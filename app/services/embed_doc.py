
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
