import pandas as pd
import csv

def parse_reviews_txt_to_csv(input_file):
    """
    Parses a text file with product reviews and writes the data to a CSV file.
    
    Args:
        input_file (str): The path to the input text file.
    """

    reviews = []  # represents whole csv

    # we open and read the txt file line by line
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            if line:
                # Split the line into review text and label
                review_text = line[:-1].strip()  # All characters except the last one
                label = line[-1]  # The last character
                reviews.append({'reviewText': review_text, 'label': label})

    # write the collected data to a csv file
    output_file = 'parsed_input_file2.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['reviewText', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # DictWriter is used to write dictionaries to CSV
        
        writer.writeheader()  # Write the header row
        for review in reviews:
            writer.writerow(review)  # Write each review to the CSV

    print(f"CSV file '{output_file}' has been created.")
    return output_file

parse_reviews_txt_to_csv('raw/amazon_cells_labelled.txt')