---

I published a write-up of this analysis on Medium.

👉 (Inferential Analysis Insights on Article Features and Popularity: Technology vs. Entertainment)[https://medium.com/@canokten/inferential-analysis-insights-on-article-features-and-popularity-technology-vs-entertainment-c4531327fb42]

This article explains the methodology (Negative Binomial Regression and GAMs), results, and key takeaways from the project in a more accessible format. 

---

# Inference on News Articles Data

## Project Overview

This project analyzes the factors that contribute to the popularity of online news articles published by Mashable, with a specific focus on identifying statistically significant predictors of article popularity. The analysis further investigates how these predictors vary across different content categories—specifically between articles in the Technology and Entertainment channels.

## Research Objectives

### 1. Feature Importance:
Identify which article-level features (e.g., keyword statistics, sentiment polarity, publication time) are most strongly associated with the number of shares an article receives.

### 2. Channel-Specific Differences:
Examine how key predictors of popularity differ between Technology and Entertainment articles. Are certain features more or less influential depending on the channel?

## Credits (Dataset Information)

Source: UCI Machine Learning Repository – "Online News Popularity"

DOI: https://doi.org/10.24432/C5NS3V

Original Publisher: Mashable (www.mashable.com)

Authors: Kelwin Fernandes, Pedro Vinagre, Paulo Cortez, Pedro Sernadela

Acquisition Date: January 8, 2015

Format: Tabular (61 features + 1 target)

Target Variable: shares (number of article shares)

The articles were published by Mashable (www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. The original content be publicly accessed and retrieved using the provided urls.

### Variable Descriptions:

The dataset contains metadata, content statistics, and sentiment metrics for nearly 40,000 Mashable articles. It does not include full article text, but includes a variety of features derived from content and metadata (e.g., token counts, LDA topics, sentiment scores, publication date, etc.). The full article content can be accessed via the original URLs provided in the data.

- _url_: URL of the article (non-predictive)
- _timedelta_: Days between the article publication and the dataset acquisition (non-predictive)
- _n_tokens_title_: Number of words in the title
- _n_tokens_content_: Number of words in the content
- _n_unique_tokens_: Rate of unique words in the content (rate 1 - 10)
- _n_non_stop_words_: Rate of non-stop words in the content (rate 1 - 10)
- _n_non_stop_unique_tokens_: Rate of unique non-stop words in the content (rate 1 - 10)
- _num_hrefs_: Number of links
- _num_self_hrefs_: Number of links to other articles published by Mashable
- _num_imgs_: Number of images
- _num_videos_: Number of videos
- _average_token_length_: Average length of the words in the content
- _num_keywords_: Number of keywords in the metadata
- _data_channel_is_lifestyle_: Is data channel 'Lifestyle'? (Yes is 1, No is 0)
- _data_channel_is_entertainment_: Is data channel 'Entertainment'? (Yes is 1, No is 0)
- _data_channel_is_bus_: Is data channel 'Business'? (Yes is 1, No is 0)
- _data_channel_is_socmed_: Is data channel 'Social Media'? (Yes is 1, No is 0)
- _data_channel_is_tech_: Is data channel 'Tech'? (Yes is 1, No is 0)
- _data_channel_is_world_: Is data channel 'World'? (Yes is 1, No is 0)
- _kw_min_min_: Number of worst keywords in the artile with min shares
- _kw_max_min_: Number of worst keywords in the artile with max shares
- _kw_avg_min_: Number of worst keywords in the artile with avg shares
- _kw_min_max_: Best keyword (min. shares)
- _kw_max_max_: Best keyword (max. shares)
- _kw_avg_max_: Best keyword (avg. shares)
- _kw_min_avg_: Avg. keyword (min. shares)
- _kw_max_avg_: Avg. keyword (max. shares)
- _kw_avg_avg_: Avg. keyword (avg. shares)
- _self_reference_min_shares_: Min. shares of referenced articles in Mashable
- _self_reference_max_shares_: Max. shares of referenced articles in Mashable
- _self_reference_avg_sharess_: Avg. shares of referenced articles in Mashable
- _weekday_is_monday_: Was the article published on a Monday? (Yes is 1, No is 0)
- _weekday_is_tuesday_: Was the article published on a Tuesday? (Yes is 1, No is 0)
- _weekday_is_wednesday_: Was the article published on a Wednesday? (Yes is 1, No is 0)
- _weekday_is_thursday_: Was the article published on a Thursday? (Yes is 1, No is 0)
- _weekday_is_friday_: Was the article published on a Friday? (Yes is 1, No is 0)
- _weekday_is_saturday_: Was the article published on a Saturday? (Yes is 1, No is 0)
- _weekday_is_sunday_: Was the article published on a Sunday? (Yes is 1, No is 0)
- _is_weekend_: Was the article published on the weekend? (Yes is 1, No is 0)
- _LDA_00_: Closeness to LDA topic 0
- _LDA_01_: Closeness to LDA topic 1
- _LDA_02_: Closeness to LDA topic 2
- _LDA_03_: Closeness to LDA topic 3
- _LDA_04_: Closeness to LDA topic 4
- _global_subjectivity_: Text subjectivity (rate from 0 to 1)
- _global_sentiment_polarity_: Text sentiment polarity (rate from -1 to 1)
- _global_rate_positive_words_: Rate of positive words in the content (rate from 0 to 1)
- _global_rate_negative_words_: Rate of negative words in the content (rate from 0 to 1)
- _rate_positive_words_: Rate of positive words among non-neutral tokens (rate from 0 to 1)
- _rate_negative_words_: Rate of negative words among non-neutral tokens (rate from 0 to 1)
- _avg_positive_polarity_: Avg. polarity of positive words (from 0 to 1)
- _min_positive_polarity_: Min. polarity of positive words (from 0 to 1)
- _max_positive_polarity_: Max. polarity of positive words (from 0 to 1)
- _avg_negative_polarity_: Avg. polarity of negative words (from 0 to 1)
- _min_negative_polarity_: Min. polarity of negative words (from 0 to 1)
- _max_negative_polarity_: Max. polarity of negative words (from 0 to 1)
- _title_subjectivity_: Title subjectivity (from 0 to 1)
- _title_sentiment_polarity_: Title polarity (from -1 to 1)
- _abs_title_subjectivity_: Absolute subjectivity level (from 0 to 1)
- _abs_title_sentiment_polarity_: Absolute polarity level(from 0 to 1)
- _shares_: Number of shares (target)

**Notes on Usage**
1. Non-Predictive Features like url and timedelta are excluded from modeling.
2. Articles are tagged with binary indicators for channel (e.g., data_channel_is_entertainment), which support subgroup comparisons.
