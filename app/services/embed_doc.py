def create_context_prompt(document_content, chunk_text):
    """
    Creates a well-structured prompt for context generation.
    """

    prompt_template = """Here is the chunk we want to situate within the whole document 
<document>
{document}
</document>

<chunk_to_analyze>
{chunk}
</chunk_to_analyze>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
."""

    return prompt_template.format(
        document=document_content.strip(), chunk=chunk_text.strip()
    )


def source_clean_string(list_source):
    if len(list_source["context"]) == 0:
        return ""

    list_context = []
    sources = "\n\n--\n Sources: \n"

    # Loop over each dictionary in the list_source
    for context in list_source["context"]:
        if "url" in context.metadata and "name" in context.metadata:
            if (
                len(context.metadata["name"]) > 50
            ):  # Issue in database, just temporary filter
                file_name = context.metadata["source"][:-4]
            else:
                file_name = context.metadata["name"]

            list_context.append(
                {"url": context.metadata["url"].split(" ")[0], "name": file_name}
            )

    # Remove duplicates by converting to a set of tuples and back to a list of dicts
    unique_data = [dict(t) for t in set(tuple(d.items()) for d in list_context)]

    # Build the sources string
    for context_metadata in unique_data:
        sources += (
            "[" + context_metadata["name"] + "](" + context_metadata["url"] + ")\n"
        )

    return sources
