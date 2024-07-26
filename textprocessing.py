import re
from indicnlp.tokenize import sentence_tokenize
from transformers import pipeline

# Load the question generation model
question_generator = pipeline("question-generation", model="ramsrigouthamg/t5-small-question-generation")


def clean_text(text):
    # Remove any unwanted characters or formatting
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()


def generate_qa_pairs(text):
    # Clean the text
    cleaned_text = clean_text(text)

    # Tokenize into sentences
    sentences = sentence_tokenize.sentence_split(cleaned_text, lang='ml')

    qa_pairs = []

    for sentence in sentences:
        # Generate questions for each sentence
        questions = question_generator(sentence, max_length=64, num_return_sequences=1)

        for q in questions:
            qa_pairs.append({
                'question': q['question'],
                'answer': sentence
            })

    return qa_pairs


# Read the text file
with open('1.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Generate QA pairs
qa_pairs = generate_qa_pairs(content)

# Print the results
for pair in qa_pairs:
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer']}")
    print()