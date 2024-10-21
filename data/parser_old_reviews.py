import pandas as pd
import csv

def parse_reviews_txt_to_csv():
    """
    Parses a text file with product reviews and writes the data to a CSV file.
    
    Args:
        input_file (str): The path to the input text file.
        output_file (str): The path to the output CSV file.
    """

    reviews = [] # represents whole csv
    review = {} # dictionary, represents one row

    # we open and read the csv line by line
    with open('raw/Arts.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            if line:
                if ': ' in line: # if one line contains a ':', then we want to split it into key and value
                    # split the line on ': ' to separate the field name and value
                    key, value = line.split(': ', 1)
                    # Check which field it is and store the corresponding value in the review dictionary
                    if key == 'product/productId':
                        review['productId'] = value
                    elif key == 'product/title':
                        review['productTitle'] = value
                    elif key == 'review/score':
                        review['productScore'] = value
                    elif key == 'review/summary':
                        review['reviewSummary'] = value
                    elif key == 'review/text':
                        review['reviewText'] = value
            else:
                # if an empty line is found, it indicates the end of a review block
                if review:
                    reviews.append(review)
                    review = {}  # resetting the review dictionary for the next entry

    # Add the last review if the file doesn't end with an empty line
    if review: # != {}
        reviews.append(review)

    # write the collected data to a csv file
    with open('parsed_input_file.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['productId', 'productTitle', 'productScore', 'reviewSummary', 'reviewText']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames) # DictWriter is used to write dictionaries to CSV
        
        writer.writeheader()  # Write the header row
        for review in reviews:
            writer.writerow(review)  # Write each review to the CSV

    print("CSV file 'parsed_input_file.csv' has been created.")
    return 'parsed_input_file.csv'

# get filepath to parsed file
processed_file_path = parse_reviews_txt_to_csv()
