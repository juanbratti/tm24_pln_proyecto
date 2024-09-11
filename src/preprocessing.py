import csv # this is for the creation of the resulting CSV file
import pandas as pd

def parse_reviews_to_csv(input_file):
    """
    Parses a text file with product reviews and writes the data to a CSV file.
    
    Args:
        input_file (str): The path to the input text file.
        output_file (str): The path to the output CSV file.
    """

    reviews = [] # represents whole csv
    review = {} # dictionary, represents one row

    # we open and read the csv line by line
    with open('../data/Arts.txt', 'r') as file:
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
    with open('../data/parsed_input_file.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['productId', 'productTitle', 'productScore', 'reviewSummary', 'reviewText']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames) # DictWriter is used to write dictionaries to CSV
        
        writer.writeheader()  # Write the header row
        for review in reviews:
            writer.writerow(review)  # Write each review to the CSV

    print("CSV file 'parsed_input_file.csv' has been created.")

# Now, we want to get the amount of reviews per productId using pandas

def get_reviews_per_product(input_file):
    """
    From a parsed csv file, returns the amount of reviews per productId (in a Series).
    
    Args:
        input_file (str): The path to the input text file.

    Returns: 
        reviews_per_product (pd.Series): The amount of reviews per productId.
    """

    # load the csv into a pandas df
    df = pd.read_csv(input_file)

    reviews_per_product = df.groupby('productId').size()
    return reviews_per_product

# Now, we want the product with the max amount of reviews

def get_product_with_max_reviews(input_file):

    """
    From a parsed csv file, returns the productId that has the most reviews.
    
    Args:
        input_file (str): The path to the input text file.

    Returns: 
        product_id (str): The productId with the most reviews.
        max_reviews (int): The amount of reviews for the product with the most reviews.
    """

    reviews_per_product = get_reviews_per_product(input_file)
    max_reviews = reviews_per_product.max()
    product_id = reviews_per_product.idxmax()

    return product_id, max_reviews


# let's parse the txt file 
# parse_reviews_to_csv('../data/Arts.txt')
reviews = pd.read_csv('../data/parsed_input_file.csv')

# let's get a dataframe that contains the information for the product_id with the most reviews
product_id, max_reviews = get_product_with_max_reviews('../data/parsed_input_file.csv')

filtered_reviews = reviews[reviews['productId'] == product_id]

print(filtered_reviews.head())







