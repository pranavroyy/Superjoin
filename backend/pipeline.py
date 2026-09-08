from . import extract, facts, db, candidates, compare


def process_pdf(pdf_path, filename, pages_per_chunk=3):
    pages = extract.extract_pages(pdf_path)
    title = extract.guess_title(pdf_path, pages)
    document_id = db.insert_document(filename, title, len(pages))

    chunks = extract.chunk_pages(pages, pages_per_chunk=pages_per_chunk)

    new_fact_ids = []
    for chunk in chunks:
        extracted, error = facts.extract_facts_from_chunk(
            chunk["text"], chunk["start_page"], chunk["end_page"]
        )
        if error:
            db.insert_issue(
                document_id,
                chunk["start_page"],
                f"LLM extraction failed for pages {chunk['start_page']}-{chunk['end_page']}: {error}",
                chunk["text"][:500],
            )
            continue

        for item in extracted:
            page = item.get("page")
            if not isinstance(page, int) or not (chunk["start_page"] <= page <= chunk["end_page"]):
                # Model hallucinated a page number outside the chunk it was given.
                # Flag it rather than silently trusting it.
                db.insert_issue(
                    document_id,
                    chunk["start_page"],
                    f"Fact for subject='{item.get('subject')}' had an out-of-range or "
                    f"missing page number ({page}); clamped to chunk start.",
                    item.get("quote", ""),
                )
                page = chunk["start_page"]

            fid = db.insert_fact(
                document_id=document_id,
                page=page,
                subject=item.get("subject", "").strip(),
                attribute=item.get("attribute", "").strip(),
                value_raw=str(item.get("value_raw", "")).strip(),
                value_number=item.get("value_number"),
                unit=item.get("unit"),
                period=item.get("period"),
                scope=item.get("scope"),
                quote=item.get("quote", "").strip(),
                fact_type=item.get("fact_type"),
            )
            new_fact_ids.append(fid)

    new_facts = [db.get_fact(fid) for fid in new_fact_ids]
    existing_facts = db.all_facts(exclude_document_id=document_id)

    relationships_created = 0
    if existing_facts and new_facts:
        pairs = candidates.find_candidates(new_facts, existing_facts)
        if pairs:
            classifications = compare.classify_pairs(pairs)
            for (fa, fb, sim), result in zip(pairs, classifications):
                if result["relationship"] == "unrelated":
                    continue
                db.insert_relationship(
                    fa["id"], fb["id"], result["relationship"],
                    result["explanation"], result["confidence"],
                )
                relationships_created += 1

    return {
        "document_id": document_id,
        "title": title,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "num_facts_extracted": len(new_fact_ids),
        "num_relationships_found": relationships_created,
    }
