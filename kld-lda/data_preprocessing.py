import re

def preprocess_data(file_path):
    """
        Preprocesses data from a story file by extracting story text and reference summary.

        Args:
        - file_path (str): Path to the story file.

        Returns:
        - tuple: A tuple containing:
            - str: The preprocessed story text.
            - str: The reference summary extracted from highlights.
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        story = file.read()

    story_text, highlights = story.split('@highlight', 1)
    story_text = re.sub(r'\s+', ' ', story_text).strip()

    highlights_list = []
    for highlight in highlights.split('@highlight'):
        cleaned_highlight = re.sub(r'\s+', ' ', highlight).strip()
        highlights_list.append(cleaned_highlight)

    reference_summary = '. '.join(highlights_list)

    return story_text, reference_summary.strip()
