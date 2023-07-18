from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import (HoverTool, Select, DataTable, TableColumn, ColumnDataSource,
                          NumeralTickFormatter, Div, StringFormatter, LogColorMapper,
                          Button, LabelSet, Title, Slider, CustomJS, Legend, TabPanel, Tabs)
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.sampledata.us_states import data as states
from bokeh.palettes import Magma256, Spectral6, Spectral10, Category10
from bokeh.io import show, output_file
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot

import numpy as np
import pandas as pd
import pickle
import math
import re

import geopandas as gpd
from shapely.geometry import MultiPolygon

def get_dataframe_by_word(word, df): #dataframe to filter for their job titles
    pattern = re.compile(r'\b{}\b'.format(word), re.IGNORECASE)
    matches = df['Job Title'].str.contains(pattern, na=False)
    filtered_df = df[matches]
    filtered_df = filtered_df[~filtered_df["Job Title"].str.contains(r'director', case=False)]
    return filtered_df

def get_dataframecomp_by_word(word, df): #dataframe to filter for the company
    pattern = re.compile(r'\b{}\b'.format(word), re.IGNORECASE)
    matches = df['Employer'].str.contains(pattern, na=False)
    filtered_df = df[matches]
    return filtered_df

def get_company_frequency_by_year(df):
    company_year_counts = df.groupby(by=["Year"], dropna=False).Employer.count()
    return company_year_counts.iloc[-6:]

def get_dataframe_by_word(word, df):
    pattern = re.compile(r'\b{}\b'.format(word), re.IGNORECASE)
    matches = df['Job Title'].str.contains(pattern, na=False)
    filtered_df = df[matches]
    filtered_df = filtered_df[~filtered_df["Job Title"].str.contains(r'director', case=False)]
    return filtered_df

def get_filtered_dataframe(df: pd.DataFrame, keyword1: str, keyword2: str) -> pd.DataFrame:
    # create regex patterns
    keyword1_pattern = re.compile(keyword1, re.IGNORECASE)
    keyword2_pattern = re.compile(keyword2, re.IGNORECASE)

    # lower-case dataframe columns for comparison
    df_lower = df.applymap(lambda s:s.lower() if type(s) == str else s)

    # filter rows based on whether they contain either keyword
    matches_keyword1 = df_lower['Employer'].apply(lambda x: bool(keyword1_pattern.search(x)) if isinstance(x, str) else False) | df_lower['Job Title'].apply(lambda x: bool(keyword1_pattern.search(x)) if isinstance(x, str) else False)
    matches_keyword2 = df_lower['Employer'].apply(lambda x: bool(keyword2_pattern.search(x)) if isinstance(x, str) else False) | df_lower['Job Title'].apply(lambda x: bool(keyword2_pattern.search(x)) if isinstance(x, str) else False)

    # return slice of original df where either conditions is True
    return df[matches_keyword1 & matches_keyword2]

def plot_company_frequencies(companies_data): #function to plot graph 4
    # Create a DataFrame where each row represents a year-company pair
    data = pd.DataFrame(columns=["year", "company", "freq"])

    for company_name, company_frequency in companies_data.items():
        df = pd.DataFrame({
            'year': company_frequency.index,
            'company': company_name,
            'freq': company_frequency.values
        })
        data = data.append(df)

    # Create a dictionary to assign a different color to each company
    companies = list(companies_data.keys())
    colors = [Spectral10[i % 10] for i in range(len(companies))]
    color_map = dict(zip(companies, colors))

    # Convert the years to strings (Bokeh uses strings for categorical axes)
    data['year'] = data['year'].astype(str)

    # Set up the grid
    grid = []
    for year in sorted(data['year'].unique()):
        year_data = data[data['year'] == year]

        # Prepare the data
        source = ColumnDataSource(year_data)

        # Set up the plot
        p = figure(x_range=companies, height=450,
                   title=f"Hiring Frequency in {year}",
                   tools="xpan,xwheel_zoom,xbox_zoom,reset", toolbar_location=None)
        # Draw bars
        bars = p.vbar(x='company', top='freq', width=0.9, source=source,
                      color=factor_cmap('company', palette=colors, factors=companies))

        # Add tools and annotations
        p.add_tools(HoverTool(tooltips=[("Company", "@company"), ("Frequency", "@freq")]))

        labels = LabelSet(x='company', y='freq', text='freq', level='glyph',
                          x_offset=-13.5, y_offset=0, source=source)
        p.add_layout(labels)

        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = math.pi / 4  # Rotate labels 45 degrees
        p.legend.orientation = "horizontal"
        p.legend.location = "top_center"

        grid.append(p)

    # Arrange plots in a grid
    grid = gridplot(grid, ncols=2)

    return grid

def get_top_10_cities_by_year(df):  # function to generate city column for dataframe
    df["City"] = df["Location"].str.split(',').str[0]
    df['Year'] = pd.to_datetime(df['Start Date']).dt.year
    top_10_cities_by_year = (
        df.groupby(['Year', 'City'])['Base Salary']
        .mean()
        .groupby('Year')
        .nlargest(10)
    )
    return top_10_cities_by_year

def plot_top_10_cities_by_year_bokeh(df):
    df['Year'] = pd.to_datetime(df['Start Date']).dt.year
    top_10_cities_by_year = (
        df.groupby(['Year', 'City'])['Base Salary']
        .mean()
        .groupby('Year')
        .nlargest(10)
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    top_10_cities_by_year = top_10_cities_by_year[top_10_cities_by_year <= 1e7]

    years = top_10_cities_by_year.index.get_level_values('Year').unique()
    num_years = len(years)
    num_plots_per_row = 2
    num_rows = int(math.ceil(num_years / num_plots_per_row))

    grid = []

    for i in range(num_years):
        year = years[i]
        if year != 2011:
            top_10_cities = top_10_cities_by_year.loc[year]

            source = ColumnDataSource(data=dict(
                city=top_10_cities.index,
                salary=top_10_cities.values,
                mean_salary=[f"${salary:.2f}" for salary in top_10_cities.values],  # Format mean salary as currency
            ))

            p = figure(x_range=list(top_10_cities.index), height=600, width=600,
                       title=f"Top 10 Cities with Highest Mean Salary - {year}",
                       tools="xpan,xwheel_zoom,xbox_zoom,reset", toolbar_location=None)

            vbar = p.vbar(x='city', top='salary', width=0.5, source=source, color=Category10[10][i % 10])

            # Create the legend, attach it to the vbar, and then set its properties
            p.add_layout(Legend(items=[("City", [vbar])]), 'right')
            # p.legend.orientation = "horizontal"
            # p.legend.location = "top_center"

            # Add text annotations for mean salary
            p.text(x='city', y='salary', text='mean_salary', text_baseline='middle',
                   text_font_size='7pt', text_color='black', source=source, angle=math.pi / 6)

            p.add_tools(HoverTool(tooltips=[("City", "@city"), ("Mean Salary", "@mean_salary")]))
            p.xaxis.major_label_orientation = math.pi / 4  # Rotate labels 45 degrees
            p.xgrid.grid_line_color = None
            p.y_range.start = 0
            p.yaxis.formatter = NumeralTickFormatter(format="0")  # display full numbers on y-axis

            grid.append(p)

    grid = gridplot(grid, ncols=num_plots_per_row)
    return grid

def plot_mean_base_salary_bokeh(data): #function to plot graph 6
    """
    Plots a grid of bar plots for each year from 2018 to 2023, showing the mean base salary using Bokeh.

    Args:
        data (list): A list of dictionaries containing the mean base salary data for each year.
                     Each dictionary should have keys 'Year', 'Category' and 'Mean Salary'.
    """
    years = ['2018', '2019', '2020', '2021', '2022', '2023']
    categories = ['Technology', 'Investment Management']
    plots = []

    for year in years:
        year_data = [d for d in data if d['Year'] == year]
        # Convert 'Mean Salary' values to strings
        for d in year_data:
            d['Mean Salary str'] = str(round(d['Mean Salary'], 2))

        # Convert list of dictionaries to pandas DataFrame
        df_year_data = pd.DataFrame(year_data)

        source = ColumnDataSource(df_year_data)

        p = figure(x_range=categories, height=250, title=f"Mean Base Salary of Data Scientist jobs for Year {year}")
        p.vbar(x='Category', top='Mean Salary', width=0.9, source=source,
               line_color='white', fill_color=factor_cmap('Category', palette=Spectral6, factors=categories))

        # Add labels
        labels = LabelSet(x='Category', y='Mean Salary', text='Mean Salary str', level='glyph',
                          x_offset=-13.5, y_offset=0, source=source)
        p.add_layout(labels)

        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Category", "@Category"),
            ("Mean Salary", "@{Mean Salary}{$0,0.00}")
        ])
        p.add_tools(hover)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.y_range.end = max([d['Mean Salary'] for d in year_data]) * 1.1  # add 10% margin at top
        p.yaxis.formatter = NumeralTickFormatter(format="0")  # display full numbers on y-axis

        plots.append(p)

    grid = gridplot(plots, ncols=2)
    output_file("salary_bars.html")
    return grid  # return the grid object
############################################CODE-STARTS-HERE############################################################

# Specify the path to your pickle file
pickle_file_path = "combined_data_finalunpurez.pkl"

# Open the pickle file in read mode
with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

# Convert Base Salaries from str to Int
data['Base Salary'] = pd.to_numeric(data['Base Salary'].str.replace(',', '')).astype('Int64')
data['Year'] = pd.to_datetime(data['Start Date']).dt.year

goldman_df = get_dataframecomp_by_word("Goldman Sachs",data)
boa_df = get_dataframecomp_by_word("bank of america",data)
cs_df = get_dataframecomp_by_word("credit suisse",data)
jpmorgan_df = get_dataframecomp_by_word("jpmorgan",data)
google_df = get_dataframecomp_by_word("google",data)
amazon_df = get_dataframecomp_by_word("amazon",data)
facebook_df = get_dataframecomp_by_word("facebook",data)
meta_df = get_dataframecomp_by_word("meta",data)
apple_df = get_dataframecomp_by_word("apple",data)
netflix_df = get_dataframecomp_by_word("netflix",data)
microsoft_df = get_dataframecomp_by_word("microsoft",data)
facemeta_df = pd.concat([facebook_df,meta_df]) #facebook changed their name to meta in 2021

goldman_counts = get_company_frequency_by_year(goldman_df)
boa_counts = get_company_frequency_by_year(boa_df)
cs_counts = get_company_frequency_by_year(cs_df)
jpmorgan_counts = get_company_frequency_by_year(jpmorgan_df)
netflix_counts = get_company_frequency_by_year(netflix_df)
google_counts = get_company_frequency_by_year(google_df)
amazon_counts = get_company_frequency_by_year(amazon_df)
facemeta_counts = get_company_frequency_by_year(facemeta_df)
microsoft_counts = get_company_frequency_by_year(microsoft_df)
apple_counts = get_company_frequency_by_year(apple_df)

SE_df = get_dataframe_by_word("Software Engineering",data)
risk_df = get_dataframe_by_word("Risk analyst",data)
trader_df = get_dataframe_by_word("Trader",data)
datasci_df = get_dataframe_by_word("Data Scientist",data)
quant_df = get_dataframe_by_word("Quantitative analyst",data)
quantdev_df = get_dataframe_by_word("Quantitative Developer",data)

datasci_invest = ['BANK OF AMERICA NA', 'BARCLAYS CAPITAL INC', 'CITADEL AMERICAS LLC','JPMORGAN CHASE & CO',
'STATE STREET BANK AND TRUST COMPANY']
datasci_tech = ['APPLE INC', 'META PLATFORMS INC', 'AMAZON WEB SERVICES INC','GOOGLE INC', 'GOOGLE LLC' 'NETFLIX INC']

amaz_ds = get_filtered_dataframe(datasci_df,"amaz","scientist")
apple_ds = get_filtered_dataframe(datasci_df,"apple","scientist")
google_ds = get_filtered_dataframe(datasci_df,"google","scientist")
meta_ds  = get_filtered_dataframe(datasci_df,"meta","scientist")
facebook_ds  = get_filtered_dataframe(datasci_df,"facebook","scientist")
netflix_ds =  get_filtered_dataframe(datasci_df,"netflix","scientist")
boa_ds = get_filtered_dataframe(datasci_df,"bank of america","scientist")
barclays_ds = get_filtered_dataframe(datasci_df,"barclays","scientist")
citadel_ds = get_filtered_dataframe(datasci_df,"citadel","scientist")
jpmorgan_ds = get_filtered_dataframe(datasci_df,"jp morgan","scientist")
statestreet_ds = get_filtered_dataframe(datasci_df,"state_street","scientist")
facemeta_ds = pd.concat([meta_ds,facebook_ds])

datasci_tech = pd.concat([amaz_ds,apple_ds,google_ds,facemeta_ds,netflix_ds])
datasci_invest = pd.concat([boa_ds,barclays_ds,citadel_ds,jpmorgan_ds,statestreet_ds])

tech_data = datasci_tech.groupby(by=["Year"], dropna=False)["Base Salary"].mean()
invest_data = datasci_invest.groupby(by=["Year"], dropna=False)["Base Salary"].mean()

avg_SE = SE_df.groupby(by=["Year"], dropna=False)["Base Salary"].mean()
avg_quant = quant_df.groupby(by=["Year"], dropna=False)["Base Salary"].mean()
avg_quantdev = quantdev_df.groupby(by=["Year"], dropna=False)["Base Salary"].mean()
avg_risk = risk_df.groupby(by=["Year"], dropna=False)["Base Salary"].mean()
avg_trader = trader_df.groupby(by=["Year"], dropna=False)["Base Salary"].mean()
avg_data = datasci_df.groupby(by=["Year"], dropna=False)["Base Salary"].mean()

data['Location_'] = data[~data['Location'].str[-2:].str.contains(',')]['Location'].str[-2:]

# Calculate the interquartile range (IQR)
Q1 = np.percentile(data['Base Salary'], 25)
Q3 = np.percentile(data['Base Salary'], 75)
IQR = Q3 - Q1

# Define the upper and lower bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
filtered_data = data[(data['Base Salary'] >= lower_bound) & (data['Base Salary'] <= upper_bound)]

# Reset the index of filtered_data and add the index as a column
filtered_data.reset_index(drop=True, inplace=True)
filtered_data['Index'] = filtered_data.index

# Create a DataTable for the filtered data with the index displayed as a column
filtered_columns = [TableColumn(field=column_name, title=column_name) for column_name in filtered_data.columns]
filtered_dataframe = DataTable(columns=filtered_columns, source=ColumnDataSource(filtered_data))

# Create a Bokeh figure for the histogram
histogram_figure = figure(title='Base Salary Histogram', background_fill_color="#fafafa")

# Define the range and initial value for the slider
bins_slider = Slider(title='Number of Bins', start=1, end=500, step=1, value=20)

# Define the hover tooltip template
hover_tooltips = [
    ("Count", "@top"),
    ("Range", "@left{0,0} to @right{0,0}")
]

# Create the HoverTool and set the tooltips
hover_hist = HoverTool(tooltips=hover_tooltips)

def update_histogram():
    # Get the current number of bins from the slider
    num_bins = bins_slider.value

    # Calculate the range and width for the histogram
    data_range = filtered_data['Base Salary'].max() - filtered_data['Base Salary'].min()
    bin_width = data_range / num_bins

    # Calculate the updated edges for the histogram
    edges = np.arange(filtered_data['Base Salary'].min(), filtered_data['Base Salary'].max() + bin_width, bin_width)

    # Create a new histogram based on the updated edges
    hist, _ = np.histogram(filtered_data['Base Salary'], bins=edges)

    # Clear the previous glyphs from the histogram figure
    histogram_figure.renderers = []

    # Add the new histogram bars
    histogram_figure.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='skyblue', line_color='black')

    # Format the x-axis tick labels to show full numbers
    histogram_figure.xaxis.formatter = NumeralTickFormatter(format='0,0')

    # Format the y-axis tick labels to show full numbers
    histogram_figure.yaxis.formatter = NumeralTickFormatter(format='0,0')

    # Add the HoverTool to the histogram figure
    histogram_figure.add_tools(hover_hist)


# Update the histogram when the slider value changes
bins_slider.on_change('value', lambda attr, old, new: update_histogram())

# Create the layout with the slider and histogram figure
layout = column(bins_slider, histogram_figure)

# Update the histogram initially
update_histogram()

# Create a box plot
boxplot_figure = figure(title='Base Salary Box Plot', background_fill_color="#fafafa")

# Calculate the box plot statistics
q1 = np.percentile(filtered_data['Base Salary'], 25)
q2 = np.percentile(filtered_data['Base Salary'], 50)
q3 = np.percentile(filtered_data['Base Salary'], 75)
lower_whisker = filtered_data['Base Salary'].min()
upper_whisker = filtered_data['Base Salary'].max()
outliers = filtered_data[(filtered_data['Base Salary'] < lower_whisker) | (filtered_data['Base Salary'] > upper_whisker)]['Base Salary']

# Add the box plot
boxplot_figure.segment([1, 1], [lower_whisker, q1], [1, 1], [q2, q3], line_color='black')
boxplot_figure.vbar(1, 0.7, q2, q3, fill_color='skyblue', line_color='black')
boxplot_figure.vbar(1, 0.7, q1, q2, fill_color='white', line_color='black')
boxplot_figure.circle(1, outliers, size=6, color='red', fill_alpha=0.6)
boxplot_figure.yaxis.formatter = NumeralTickFormatter(format='0,0')

# Add hover tooltips to the box plot
hover_boxplot = HoverTool(tooltips=[("Lower Whisker", str(lower_whisker)), ("Q1", str(q1)),
                                    ("Median", str(q2)), ("Q3", str(q3)), ("Upper Whisker", str(upper_whisker))])
boxplot_figure.add_tools(hover_boxplot)

########################################################################################################################
# List of salary ranges for filtering
salary_ranges = ['All', 'Below 100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k',
                 '500k-600k', '600k-700k', '700k-800k', 'Above 800k']

# Create a Select widget for salary filtering
salary_filter_select = Select(title="Filter by Base Salary", options=salary_ranges, value=salary_ranges[0])

def salary_filter_callback(attr, old, new):
    selected_salary_range = salary_filter_select.value

    filtered_data = data.copy()  # Create a copy of the original data
    if selected_salary_range == 'Below 100k':
        filtered_data = filtered_data[filtered_data['Base Salary'] < 100000]
    elif selected_salary_range == 'Above 800k':
        filtered_data = filtered_data[filtered_data['Base Salary'] >= 800000]
    elif selected_salary_range == 'All':
        pass  # Keep the data as it is
    else:
        salary_range_values = selected_salary_range.split('-')
        lower_bound = int(salary_range_values[0].replace('k', '')) * 1000
        upper_bound = int(salary_range_values[1].replace('k', '')) * 1000
        filtered_data = filtered_data[
            (filtered_data['Base Salary'] >= lower_bound) & (filtered_data['Base Salary'] < upper_bound)]

    filtered_source.data = dict(ColumnDataSource(filtered_data).data)
    filtered_dataframe.source = ColumnDataSource(filtered_data)

    update_job_list(filter_select.value)  # Update the job list based on the selected job filter

salary_filter_select.on_change('value', salary_filter_callback)


# List of interesting jobs for filtering
interesting_jobs = ['All', 'Quantitative Analyst', 'Quantitative Developer', 'Quantitative Trader',
                    'Risk Analyst', 'Software Engineer', 'Data Analyst', 'Data Scientist']

# Changing everything into capital letters
interesting_jobs = [job.upper() for job in interesting_jobs]

# Create a Select widget for job filtering
filter_select = Select(title="Filter by Job", options=interesting_jobs, value=interesting_jobs[0])

def filter_callback(attr, old, new):
    selected_job = filter_select.value
    if selected_job == 'All':
        filtered_data = data.copy()  # Create a copy of the original data
    else:
        filtered_data = data[data['Job Title'] == selected_job]

    filtered_source.data = dict(ColumnDataSource(filtered_data).data)
    filtered_dataframe.source = ColumnDataSource(filtered_data)
    update_job_list(selected_job)  # Update the job list based on the selected job filter

# Add the filter callback to the Select widget
filter_select.on_change('value', filter_callback)

# Create a DataFrame using the ColumnDataSource
filtered_source = ColumnDataSource(filtered_data)

# Create a layout for the data tab
data_stats = data.describe().reset_index()

# Convert data_stats to a Bokeh DataTable
columns = [
    TableColumn(field='index', title='Index'),
    *[
        TableColumn(field=column_name, title=column_name.capitalize())
        for column_name in data_stats.columns
        if column_name != 'index'
    ]
]

data_stats_table = DataTable(columns=columns, source=ColumnDataSource(data_stats))

# Create a Div widget for displaying statistics
stats_div = Div(text="Mean Base Salary: ${:,.2f}<br>Maximum Base Salary: ${:,}<br>Minimum Base Salary: ${:,}"
                .format(data['Base Salary'].mean(), data['Base Salary'].max(), data['Base Salary'].min()))

# Create a Div widget for displaying the list of jobs
job_list_div = Div(text="")

# Define a function to update the job list based on the selected job filter
def update_job_list(selected_job):
    if selected_job in interesting_jobs:
        job_list_div.text = "<b>All Jobs:</b> {}".format(", ".join(interesting_jobs))
    else:
        job_list_div.text = "<b>Filtered Jobs:</b> {}".format(selected_job)

# Call the update_job_list function initially to populate the Div with the full list of jobs
update_job_list(interesting_jobs[0])

# Create the reset button
reset_button = Button(label="Reset Filters")


def reset_filters():
    filter_select.value = interesting_jobs[0]  # Set the job filter back to its initial value
    salary_filter_select.value = salary_ranges[0]  # Set the salary filter back to its initial value

    # Reapply the filters and update the data table
    selected_job = filter_select.value
    filtered_data = data[data['Job Title'] == selected_job]
    filtered_source.data = dict(ColumnDataSource(filtered_data).data)
    filtered_dataframe.source = ColumnDataSource(filtered_data)
    update_job_list(selected_job)  # Update the job list based on the selected job filter

# Link the callback function to the button
reset_button.on_click(reset_filters)

# Read world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Exclude Antarctica because it takes up a lot of space
world = world[world['name'] != "Antarctica"]

# Convert Geopandas data to Bokeh-compatible format
world_xs = []
world_ys = []

for _, country in world.iterrows():
    if isinstance(country.geometry, MultiPolygon):
        # Each part of a MultiPolygon is its own separate polygon.
        for part in country.geometry.geoms:
            world_xs.append(list(part.exterior.coords.xy[0]))
            world_ys.append(list(part.exterior.coords.xy[1]))
    else:
        world_xs.append(list(country.geometry.exterior.coords.xy[0]))
        world_ys.append(list(country.geometry.exterior.coords.xy[1]))

# Prepare data for US map
grouped_data = data.groupby('Location_').size().reset_index(name='counts')
state_counts = grouped_data.set_index('Location_')['counts'].to_dict()

state_xs = [states[code]["lons"] for code in states]
state_ys = [states[code]["lats"] for code in states]
state_colors = [state_counts.get(code, 1) for code in states]
state_names = [states[code]["name"] for code in states]

# Color mapper
color_mapper = LogColorMapper(palette=Magma256)

# Prepare the data for the map
us_map_data = dict(
    xs=state_xs,
    ys=state_ys,
    color=state_colors,
    state_names=state_names,
    state_counts=[state_counts.get(code, 0) for code in states],
)

# Create a ColumnDataSource for the maps
us_map_source = ColumnDataSource(data=us_map_data)
world_map_source = ColumnDataSource(data=dict(xs=world_xs, ys=world_ys))

# Create the map figure
p = figure(title="Jobs distribution across US",
           toolbar_location="left",
           width=850, height=800)

# Plot world map
p.patches('xs', 'ys', fill_color="#ffffff", line_color="#000000", source=world_map_source)

# Plot US states with jobs
p.patches('xs', 'ys', fill_color={'field': 'color', 'transform': color_mapper},
          fill_alpha=0.7, line_color="white", line_width=0.5, source=us_map_source)

# Plot US states with jobs
us_patches = p.patches('xs', 'ys', fill_color={'field': 'color', 'transform': color_mapper},
          fill_alpha=0.7, line_color="white", line_width=0.5, source=us_map_source)

# Add a hover tool for the US states
hover_map = HoverTool(renderers=[us_patches],
                      tooltips=[("State", "@state_names"), ("Number of Jobs", "@state_counts")])
p.add_tools(hover_map)


########################################################################################################################
# Create a layout for the data tab
data_layout_left = column(
    row(filter_select, salary_filter_select, reset_button),
    Div(text="", width=800, height=20),  # Add an empty Div with appropriate width and height
    filtered_dataframe,
    data_stats_table,
    sizing_mode='stretch_width'  # Set sizing_mode to stretch_width
)

data_layout_right = p

data_layout = row(data_layout_left, data_layout_right)

data_tab = TabPanel(child=data_layout, title="Summary Data for all H1B Data")


# Create Tab 2 layout with the histogram, box plot, and stats_div
tab2_layout = column(row(layout, boxplot_figure), stats_div)
tab2 = TabPanel(child=tab2_layout, title="Visualization of Base Salary")

########################################################################################################################
# Create a data source for each job title
SE_source = ColumnDataSource(data=dict(x=avg_SE.index, y=avg_SE))
quant_source = ColumnDataSource(data=dict(x=avg_quant.index, y=avg_quant))
quantdev_source = ColumnDataSource(data=dict(x=avg_quantdev.index, y=avg_quantdev))
risk_source = ColumnDataSource(data=dict(x=avg_risk.index, y=avg_risk))
trader_source = ColumnDataSource(data=dict(x=avg_trader.index, y=avg_trader))
data_source = ColumnDataSource(data=dict(x=avg_data.index, y=avg_data))


# Create a new plot with white background
p3 = figure(title="Average Salary by Year", x_axis_label='Year', y_axis_label='Average Salary',
           background_fill_color='white', border_fill_color='white', width=1400, height=700, toolbar_location="below")

# Change the y ticker to show full numbers
p3.yaxis.formatter = NumeralTickFormatter(format="0")

# Define a color palette for the lines
palette = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

# Add a line renderer for each job title with different colors, and increase line_width
p3.line('x', 'y', source=SE_source, legend_label="Software Engineering", line_color=palette[0], line_width=2)
p3.line('x', 'y', source=quant_source, legend_label="Quantitative Analyst", line_color=palette[1], line_width=2)
p3.line('x', 'y', source=quantdev_source, legend_label="Quantitative Developer", line_color=palette[2], line_width=2)
p3.line('x', 'y', source=risk_source, legend_label="Risk Analyst", line_color=palette[3], line_width=2)
p3.line('x', 'y', source=trader_source, legend_label="Trader", line_color=palette[4], line_width=2)
p3.line('x', 'y', source=data_source, legend_label="Data Scientist", line_color=palette[5], line_width=2)

# Add a hover tool with full number display
hover = HoverTool(tooltips=[
    ("Year", "@x{0.0}"),
    ("Average Salary", "@y{$0,0}"),
])
p3.add_tools(hover)

# Change legend and axis colors to be visible on white background
p3.legend.label_text_color = 'black'
p3.xaxis.axis_label_text_color = 'black'
p3.yaxis.axis_label_text_color = 'black'
p3.xaxis.major_label_text_color = 'black'
p3.yaxis.major_label_text_color = 'black'
p3.title.text_color = 'black'

# Move the legend to the upper left corner
p3.legend.location = "top_left"

########################################################################################################################
companies_data = {
    'Goldman Sachs': goldman_counts,
    'Bank of America Merill Lynch': boa_counts,
    'Credit Suisse': cs_counts,
    'Microsoft': microsoft_counts,
    'JP Morgan Chase & Co': jpmorgan_counts,
    'Netflix': netflix_counts,
    'Google': google_counts,
    'Amazon': amazon_counts,
    'Facebook/Meta': facemeta_counts,
    'Apple': apple_counts}

# Create Panel
p4 = plot_company_frequencies(companies_data)
########################################################################################################################
top10_cities_eachyr = get_top_10_cities_by_year(data)
p5 = plot_top_10_cities_by_year_bokeh(data)

########################################################################################################################
invest_data = invest_data.loc[2018:2023]
tech_data = tech_data.loc[2018:2023]

data_for_tech_invest = [
    {'Year': '2018', 'Category': 'Technology', 'Mean Salary': tech_data.iloc[0]},
    {'Year': '2018', 'Category': 'Investment Management', 'Mean Salary': invest_data.iloc[0]},
    {'Year': '2019', 'Category': 'Technology', 'Mean Salary': tech_data.iloc[1]},
    {'Year': '2019', 'Category': 'Investment Management', 'Mean Salary': invest_data.iloc[1]},
    {'Year': '2020', 'Category': 'Technology', 'Mean Salary': tech_data.iloc[2]},
    {'Year': '2020', 'Category': 'Investment Management', 'Mean Salary': invest_data.iloc[2]},
    {'Year': '2021', 'Category': 'Technology', 'Mean Salary': tech_data.iloc[3]},
    {'Year': '2021', 'Category': 'Investment Management', 'Mean Salary': invest_data.iloc[3]},
    {'Year': '2022', 'Category': 'Technology', 'Mean Salary': tech_data.iloc[4]},
    {'Year': '2022', 'Category': 'Investment Management', 'Mean Salary': invest_data.iloc[4]},
    {'Year': '2023', 'Category': 'Technology', 'Mean Salary': tech_data.iloc[5]},
    {'Year': '2023', 'Category': 'Investment Management', 'Mean Salary': invest_data.iloc[5]}
]

p6 = plot_mean_base_salary_bokeh(data_for_tech_invest)

########################################################################################################################
# Create empty Tab 3, Tab 4, Tab 5 ,Tab 6
tab3 = TabPanel(child=p3, title="Average Salary Changes of Jobs of Interest")
tab4 = TabPanel(child=p4, title='Company Hiring Frequency from 2018 to 2023')
tab5 = TabPanel(child=p5, title="Top 10 Highest Paying Cities across the years")
tab6 = TabPanel(child=p6, title="Base Salary of Data Scientist with regards to Investment Management and Technology")

# Create a Tabs layout
tabs_layout = Tabs(tabs=[data_tab, tab2, tab3, tab4, tab5, tab6])

# Set the app layout
curdoc().add_root(tabs_layout)