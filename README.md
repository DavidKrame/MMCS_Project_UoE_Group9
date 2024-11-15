# Optimisation Model for Strategic Planning in the TV Sector

## Mathematical, Modelling and Consulting Skills – MSc Operational Research 2024/2025

### Link to the Mathematical Model in Overleaf

[Click here to view the model on Overleaf](https://www.overleaf.com/9631347139bgqygbhyzjjz#7ba809)

### 0. Project Files

This project contains several Jupyter notebooks :
- **DataAnalysis_...**: Various analyses of the datasets.
- **FirstWeek_only_Extraction**: Extracts the first week of data.
- **Vanilla_Model_1**: The optimization model implemented in Xpress-Python.

Additionally, there is a folder named **Data** containing all the relevant datasets.

### 1. Introduction

This project presents a small-scale set of competitor TV channels, exploring optimal solutions for maximizing viewership through intelligent scheduling and planning a cost-effective advertising strategy. The challenge considers four channels streaming movies all the time, with daily programming running from 7 AM until midnight and 5-minute time slots every 30 minutes.

As consultants hired by a TV channel, we received a blank timeline along with the planned schedules for three competitors. The primary objective is to deliver a systematic methodology to optimize the channel’s own viewership, retaining existing audiences and attracting new viewers, while adapting to competitors' scheduling strategies.

### 2. Key Facts About the Data

The simulated data aims to represent realistic distributions and logical relationships, although gaps and inconsistencies may exist. The following datasets are included:

- **Movie Database**: Nearly 6,000 titles from which channels can choose.
  - Movie Title
  - IMDb Average Vote (1-10)
  - Vote Count
  - Release Date
  - Box Office Revenue
  - Run Time
  - Budget
  - Genre(s)
  - Number of Ad Breaks
  - Runtime with Ads
  - Children, Adults, and Retirees Scaled Popularity

- **Channels’ Schedule Databases** (Competitor Channels 0, 1, and 2): 
  - Date-Time (5-min slots)
  - Content Type (Movie/Advert)
  - Popularity Scores and View Counts for Children, Adults, and Retirees
  - Prime Time Factor
  - Ad Slot Price

- **Channels’ Conversion Rates**: Probability that advertising on a competitor channel leads to positive conversion into your own channel.

- **Your Channel Schedule**: Baseline view counts and prime time factors for children, adults, and retirees.

### 3. Stochastic Modelling

Advertising increases viewership through probabilities, with viewership calculated based on baseline counts multiplied by popularity. The model incorporates conversion rates based on advertising placements on competitors' channels.

### 4. Key Objectives and Constraints

The main objectives are to maximize overall viewership through:
- Intelligent scheduling aligned with competitors’ schedules.
- Efficient advertising strategies using both own and competitors’ channels.

Expenses include streaming license fees and advertising costs, while revenue is driven by viewership and sold advert slots.

### 5. Timeline and Iterative Approach

The model optimizes the TV channel’s schedule and advertising over the first week, starting from 01 October 2024. An iterative approach will refine strategies based on weekly learnings and adjustments to advertising budgets.

### 6. Exploratory Questions

The executive board seeks recommendations to enhance the optimization strategy, focusing on aspects such as optimal pricing for advert slots, competitor scheduling tactics, minimum revenue thresholds, demographic trends, content strategies, and more.

### Authors
- Adeyinka Badmus
- David Krame Kadurha
- Hariaksh Pandya
- Rónán Sweeney McCarron

---