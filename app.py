import streamlit as st
import pandas as pd
import spacy
import re
from datetime import datetime
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModel
import torch
############################################################################################################
# # Load models
# nlp = spacy.load("en_core_web_sm")

# # Load the BART model and tokenizer
# bart_model_name = "facebook/bart-large-cnn"
# tokenizer = BartTokenizer.from_pretrained(bart_model_name)
# bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# # Load the FinBERT model and tokenizer
# finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# finbert_model = AutoModel.from_pretrained('yiyanghkust/finbert-tone')
###########################################################################################################

# Path to load models from Google Drive
drive_model_path = '/content/drive/My Drive/NLP_models/'

# Load models
nlp = spacy.load(drive_model_path + "en_core_web_sm")

# Load the BART model and tokenizer from Google Drive
bart_model_name = drive_model_path + "facebook-bart-large-cnn/"
tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Load the FinBERT model and tokenizer from Google Drive
finbert_tokenizer = AutoTokenizer.from_pretrained(drive_model_path + 'finbert-tone/')
finbert_model = AutoModel.from_pretrained(drive_model_path + 'finbert-tone/')

###########################################################################################################

# Streamlit app title
st.title("Financial Data Query Assistant")

# File uploader for Excel files
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Helper function to clean and convert value strings to float
    def clean_value(value):
        if isinstance(value, str):
            cleaned_value = re.sub(r'[^\d.]+', '', value)
            return float(cleaned_value) if cleaned_value else 0.0
        return float(value)

    # Function to get value based on metric and year
    def get_value(metric, year):
        result = df[(df['metrics'].str.lower() == metric.lower()) & (df['year'] == year)]
        if not result.empty:
            value = result['value'].values[0]
            return clean_value(value)
        return None

    # Function to find the year with the highest value for a metric
    def get_highest_year(metric):
        metric_df = df[df['metrics'].str.lower() == metric.lower()]
        if not metric_df.empty:
            metric_df['cleaned_value'] = metric_df['value'].apply(clean_value)
            max_value_row = metric_df.loc[metric_df['cleaned_value'].idxmax()]
            return max_value_row['year'], max_value_row['cleaned_value']
        return None, None

    # Function to calculate changes between consecutive years
    def calculate_changes(metric):
        metric_df = df[df['metrics'].str.lower() == metric.lower()]
        if not metric_df.empty:
            metric_df = metric_df.sort_values(by='year')
            metric_df['cleaned_value'] = metric_df['value'].apply(clean_value)
            metric_df['change'] = metric_df['cleaned_value'].diff()
            max_change_row = metric_df.loc[metric_df['change'].idxmax()]
            min_change_row = metric_df.loc[metric_df['change'].idxmin()]
            return (max_change_row['year'], max_change_row['change']), (min_change_row['year'], min_change_row['change'])
        return None, None

    # Function to generate a BART model response
    def generate_response(query):
        inputs = tokenizer.encode("summarize: " + query, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = bart_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Function to convert relative time expressions to specific years
    def parse_relative_time(query):
        current_year = datetime.now().year
        if "last year" in query.lower():
            return current_year - 1
        elif "this year" in query.lower():
            return current_year
        elif "next year" in query.lower():
            return current_year + 1
        elif "recently" in query.lower():
            return current_year
        return None

    # Function to find the closest matching metrics using FinBERT embeddings
    def find_closest_metrics_finbert(query_metric, metrics_list, top_n=5):
        # Tokenize and embed the query and metrics
        query_tokens = finbert_tokenizer(query_metric.lower(), return_tensors='pt')
        query_embedding = finbert_model(**query_tokens).last_hidden_state.mean(dim=1)
        
        similarities = []
        for metric in metrics_list:
            metric_tokens = finbert_tokenizer(metric, return_tensors='pt')
            metric_embedding = finbert_model(**metric_tokens).last_hidden_state.mean(dim=1)
            similarity = torch.cosine_similarity(query_embedding, metric_embedding).item()
            similarities.append((metric, similarity))
        
        # Sort by similarity and return top_n similar metrics
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [metric for metric, _ in similarities[:top_n]]

    # Function to extract and match metrics from the query
    def extract_and_match_metric(query, metrics_list):
        doc = nlp(query)
        for token in doc:
            if token.text.lower() in metrics_list:
                return token.text.lower()
        return None

    # Main function to handle queries
    def handle_query(query):
        metrics_list = df['metrics'].str.lower().unique()
        metric = extract_and_match_metric(query, metrics_list)
        year = None
        compare_year = None
        aggregate_function = None
        find_max = False
        find_change = False
        find_when_change = False
        
        # Check for change-related keywords
        if any(keyword in query.lower() for keyword in ["increase", "decrease", "difference", "change"]):
            find_change = True
            if any(keyword in query.lower() for keyword in ["most", "highest"]):
                find_when_change = "max"
            elif any(keyword in query.lower() for keyword in ["least", "lowest"]):
                find_when_change = "min"
        
        # Check for aggregate keywords
        if "total" in query.lower() or "sum" in query.lower():
            aggregate_function = "sum"
        elif "average" in query.lower() or "mean" in query.lower():
            aggregate_function = "mean"
        elif "highest" in query.lower() or "maximum" in query.lower():
            find_max = True
        
        # If the exact metric is not found, find similar metrics
        if not metric:
            similar_metrics = find_closest_metrics_finbert(query, metrics_list, top_n=3)
            if similar_metrics:
                similar_metrics_str = ', '.join(similar_metrics)
                response = (f"I couldn't find the exact metric '{query}', but here are some similar metrics you can ask about: "
                            f"{similar_metrics_str}.")
                return response
            else:
                return generate_response(query)
        
        metric_found_message = ""
        
        # Handle queries asking for the highest value
        if metric and find_max:
            year, value = get_highest_year(metric)
            if year and value is not None:
                response = f"{metric_found_message} The highest {metric.upper()} was {value:.2f} in {year}."
                return response
        
        # Handle queries asking for aggregated values
        if metric and aggregate_function:
            if aggregate_function == "sum":
                total_value = df[df['metrics'].str.lower() == metric]['value'].apply(clean_value).sum()
                response = f"{metric_found_message} The total {metric.upper()} is {total_value:.2f}."
                return response
            elif aggregate_function == "mean":
                average_value = df[df['metrics'].str.lower() == metric]['value'].apply(clean_value).mean()
                response = f"{metric_found_message} The average {metric.upper()} is {average_value:.2f}."
                return response
        
        # Handle queries asking about the largest or smallest change
        if metric and find_change:
            if find_when_change:
                max_change, min_change = calculate_changes(metric)
                if find_when_change == "max" and max_change:
                    year, change = max_change
                    response = f"{metric_found_message} The {metric.upper()} increased the most by {change:.2f} in {year}."
                    return response
                elif find_when_change == "min" and min_change:
                    year, change = min_change
                    response = f"{metric_found_message} The {metric.upper()} decreased the most by {abs(change):.2f} in {year}."
                    return response

        # Handle single year or comparison queries
        year = parse_relative_time(query)
        
        if not year:
            years = [int(token.text) for token in nlp(query) if token.ent_type_ == "DATE" and token.text.isdigit()]
            if len(years) == 2:
                year, compare_year = years
            elif len(years) == 1:
                year = years[0]
        
        if metric and year:
            value = get_value(metric, year)
            if value is not None:
                if find_change and compare_year:
                    value_compare = get_value(metric, compare_year)
                    if value_compare is not None:
                        difference = value_compare - value
                        response = (f"{metric_found_message} The {metric.upper()} was {value_compare:.2f} in {compare_year} and "
                                    f"{value:.2f} in {year}, a difference of {difference:.2f}.")
                        return response
                else:
                    response = f"{metric_found_message} The {metric.upper()} in {year} was {value:.2f}."
                    return response

        return generate_response(query)

    # Input for user queries
    user_query = st.text_input("Ask a financial data query:")
    
    if user_query:
        response = handle_query(user_query)
        st.write(response)
