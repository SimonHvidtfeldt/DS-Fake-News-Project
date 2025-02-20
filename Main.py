import re 

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespaces, tabs, and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace numbers with <NUM>
    text = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', text)

    # Replace common date formats with <DATE>
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '<DATE>', text)  # Matches formats like 12/05/2023 or 12-05-23

    # Replace emails with <EMAIL>
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)

    # Replace URLs with <URL>
    text = re.sub(r'https?://\S+', '<URL>', text)

    return text

# Example usage:
global cleaned_text_applied_ct
cleaned_text_applied_re = clean_text(file_content)
