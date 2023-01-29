import os
import string
import math
import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    # The text is defined as a dictionary
    corpus = dict()

    # Maps the filename of each text file to the file's contents as a string
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding='utf8') as file:
                corpus[filename] = file.read()

    return corpus


def tokenize(document):
    # Sets the punctuation and stop words that are separated from the words
    punctuation = string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")

    # Process document by converting all the words to lowercase
    words = nltk.word_tokenize(document.lower())

    # Removes any punctuation or English stopwords
    words = [word for word in words if word not in punctuation and word not in stop_words]

    return words


def compute_idfs(documents):
    # Dictionary of IDF values
    counts = dict()

    # Any word that appears in at least one of the documents is added to a dictionary
    for filename in documents:
        seen_words = set()
        for word in documents[filename]:
            if word not in seen_words:
                seen_words.add(word)
                try:
                    counts[word] += 1
                except KeyError:
                    counts[word] = 1

    compute_dic = {word: math.log(len(documents) / counts[word]) for word in counts}

    return compute_dic


def top_files(query, files, idfs, n):
    # Dictionary mapping names of files to a list of their words
    file_scores = dict()

    for file, words in files.items():
        total_tf_idf = 0
        for word in query:
            total_tf_idf += words.count(word) * idfs[word]
        file_scores[file] = total_tf_idf

    # List of the filenames of the 'n' top files that match the query
    ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_files = [x[0] for x in ranked_files]

    return ranked_files[:n]


def top_sentences(query, sentences, idfs, n):
    sentence_scores = list()

    # Gets the 'n' top sentences that match the query, ranked according to idf
    for sentence in sentences:
        sentence_values = [sentence, 0, 0]
        for word in query:
            if word in sentences[sentence]:
                # Compute “matching word measure”
                sentence_values[1] += idfs[word]
                # Compute "query term density"
                sentence_values[2] += sentences[sentence].count(word) / len(sentences[sentence])

        sentence_scores.append(sentence_values)

    top_list = [sentence for sentence, mwm, qtd in
                sorted(sentence_scores, key=lambda item: (item[1], item[2]), reverse=True)][:n]

    return top_list


if __name__ == "__main__":
    main()
