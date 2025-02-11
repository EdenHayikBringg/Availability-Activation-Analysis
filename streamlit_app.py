import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats

# ---------------------------
# App Title and Description
# ---------------------------
st.image('./bringg.png', width=200)
st.title("Impact of Availability Slots on Average Tasks per Route")
st.markdown("""
This dashboard compares **each month after the availability slots service activation** with 
the **same month before the activation**, while **normalizing the data for business growth**.

### Steps:
1. Upload your CSV file.
2. Select the activation date of the extra service.
3. Choose how many months after the split to include.
4. Select the teams to analyze.
""")

show_sql_query = st.checkbox("Click to show the relevant sql query for this service", value=False)
if show_sql_query:
    st.code("""
    SELECT 
        count(*) AS total_count, 
        month, 
        run_id, 
        teams_ids, 
        array_agg(id) AS task_ids
    FROM tasks 
    WHERE merchant_id = ??
    AND tasks.delete_at IS NULL
    AND tasks.task_type_id != 3
    AND month > 202305
    AND month < 202501
    AND CASE 
        WHEN cardinality(teams_ids) > 0 THEN teams_ids[1] 
        ELSE NULL 
    END IN (??,??) # team names
    GROUP BY run_id, month, teams_ids
    """)

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # ---------------------------
    # Load and Prepare Data
    # ---------------------------
    df = pd.read_csv(uploaded_file)

    # Rename _col0 to "total_count" for clarity
    if "_col0" in df.columns:
        df.rename(columns={"_col0": "total_count"}, inplace=True)

    # Convert 'month' to string and then to datetime format
    df['month'] = df['month'].astype(str)
    df['month_dt'] = pd.to_datetime(df['month'], format='%Y%m', errors='coerce')

    # Remove invalid months
    df = df.dropna(subset=['month_dt'])

    # Extract year and month
    df['year'] = df['month_dt'].dt.year
    df['month_only'] = df['month_dt'].dt.month

    # Delete 2025 rows
    df = df[df['year'] != 2025]

    # Delete runs with only 1 task
    # df = df[df['total_count'] > 1]

    # ---------------------------
    # Threshold Date Selection via UI
    # ---------------------------
    threshold_date = st.date_input("Select the activation date of the extra service", value=pd.to_datetime("2024-06-01").date())
    threshold = pd.to_datetime(threshold_date)

    # ---------------------------
    # User Input: Number of Months to Include After Split
    # ---------------------------
    max_months_available = df[df['month_dt'] >= threshold]['month_dt'].nunique()
    num_months_forward = st.number_input(
        f"Enter how many months to analyze after {threshold.strftime('%Y-%m')}:",
        min_value=1, 
        max_value=max_months_available,
        value=min(12, max_months_available)  # Default to 12 months or max available
    )

    if num_months_forward > max_months_available:
        st.warning(f"⚠️ You requested {num_months_forward} months, but only {max_months_available} months are available in the dataset after {threshold.strftime('%Y-%m')}. Adjusting to {max_months_available} months.")
        num_months_forward = max_months_available

    # Convert `run_id` to numeric and drop invalid entries
    df['run_id'] = pd.to_numeric(df['run_id'], errors='coerce')
    df = df.dropna(subset=['run_id'])
    df = df[df['run_id'] != 0]

    # Clean `teams_ids`
    df['teams_ids'] = df['teams_ids'].astype(str).str.replace(r"[\[\] ]", "", regex=True)
    df = df.assign(teams_ids=df['teams_ids'].str.split(",")).explode('teams_ids')
    df['teams_ids'] = df['teams_ids'].astype(str)


    # ---------------------------
    # Team Selection
    # ---------------------------
    unique_teams = sorted(df['teams_ids'].unique())
    selected_teams = st.multiselect("Select Team(s) to Analyze", unique_teams, default=unique_teams)
    df = df[df['teams_ids'].isin(selected_teams)]

    # ---------------------------
    # Data Normalization
    # ---------------------------
    normalize_data = st.checkbox("Normalize by business growth", value=True)

    # ---------------------------
    # Data Distribution
    # ---------------------------
    st.subheader("Data Distribution")
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Unique Run IDs: {df['run_id'].nunique()}")
    st.write(f"Avarage Tasks per Run ID: {df['total_count'].mean():.2f}")

    st.subheader("Tasks per Run ID Distribution")
    # select a month to show the distribution of tasks per run_id
    selected_month = st.selectbox("Select a month to show the distribution of tasks per run_id", df['month_dt'].dt.strftime('%Y-%m').unique())
    df_month = df[df['month_dt'].dt.strftime('%Y-%m') == selected_month]

    # general distribution of tasks per run_id

    fig_hist = px.histogram(
        df_month,
        x='total_count',
        nbins=50,
        labels={"total_count": "Tasks per Run ID"},
        title=f"Tasks per Run ID Distribution for Selected Month (Average: {df_month['total_count'].mean():.2f})"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------------
    # Compute Monthly Average Tasks per Run ID
    # ---------------------------
    df_grouped = df.groupby(['month_dt', 'year', 'month_only'], as_index=False).agg(
        total_tasks=('total_count', 'sum'),
        unique_runs=('run_id', 'nunique')
    )

    # Compute avg tasks per run_id
    df_grouped['avg_tasks_per_run'] = df_grouped['total_tasks'] / df_grouped['unique_runs']

    # ---------------------------
    # Normalize Growth by Business Expansion
    # ---------------------------
    total_tasks_per_month = df_grouped.groupby(['month_only', 'year'], as_index=False)['total_tasks'].sum()

    # Merge with previous year's data
    total_tasks_per_month['previous_year'] = total_tasks_per_month['year'] - 1
    df_growth = total_tasks_per_month.merge(
        total_tasks_per_month,
        left_on=['month_only', 'previous_year'],
        right_on=['month_only', 'year'],
        suffixes=("", "_previous"),
        how='left'
    )

    # Compute growth factor
    df_growth['growth_factor'] = df_growth['total_tasks'] / df_growth['total_tasks_previous']
    df_growth = df_growth[['month_only', 'year', 'growth_factor']]

    # Merge into main dataset
    df_grouped = df_grouped.merge(df_growth, on=['month_only', 'year'], how='left')

    # Normalize avg tasks per run_id
    df_grouped['normalized_avg_tasks_per_run'] = df_grouped['avg_tasks_per_run'] / df_grouped['growth_factor']

    # Handle NaN values in normalization
    df_grouped['normalized_avg_tasks_per_run'] = df_grouped['normalized_avg_tasks_per_run'].fillna(df_grouped['avg_tasks_per_run'])

    # ---------------------------
    # Filter Based on Selected Months After the Split
    # ---------------------------
    df_filtered = df_grouped.copy()
    df_filtered['post_split'] = df_filtered['month_dt'] >= threshold

    # Get the first `num_months_forward` months after the split
    post_split_months = df_filtered[df_filtered['post_split']].sort_values('month_dt').head(num_months_forward)['month_only'].unique()
    df_filtered = df_filtered[df_filtered['month_only'].isin(post_split_months)]

    # ---------------------------
    # Plot: Year-over-Year Comparison of Normalized Tasks per Run
    # ---------------------------
    if normalize_data:
        st.subheader("Year-over-Year Comparison of Normalized Avg Tasks per Run ID")
        fig_trend = px.line(
            df_filtered,
            x='month_only',
            y='normalized_avg_tasks_per_run',
            color='year',
            markers=True,
            labels={"month_only": "Month", "normalized_avg_tasks_per_run": "Normalized Avg Tasks per Run ID"},
            title="Same-Month Comparison Across Years"
        )
    else:
        st.subheader("Year-over-Year Comparison of Avg Tasks per Run ID")
        fig_trend = px.line(
            df_filtered,
            x='month_only',
            y='avg_tasks_per_run',
            color='year',
            markers=True,
            labels={"month_only": "Month", "avg_tasks_per_run": "Avg Tasks per Run ID"},
            title="Same-Month Comparison Across Years"
        )
    st.plotly_chart(fig_trend, use_container_width=True)


    # ---------------------------
    # Statistical Test: Paired Welch’s T-Test
    # ---------------------------
    st.subheader("Statistical Comparison: Same-Month Year-over-Year Changes")

    before_data = df_filtered[df_filtered['year'] == (threshold.year - 1)]
    after_data = df_filtered[df_filtered['year'] == threshold.year]

    paired_data = before_data.merge(
        after_data,
        on="month_only",
        suffixes=("_before", "_after")
    )

    if not paired_data.empty:
        t_stat, p_val = stats.ttest_rel(
            paired_data['normalized_avg_tasks_per_run_before'],
            paired_data['normalized_avg_tasks_per_run_after']
        )

        st.write(f"**Paired T-statistic:** {t_stat:.3f}")
        st.write(f"**P-value:** {p_val:.3e}")

        if p_val < 0.05:
            st.info("There is a **statistically significant** change in normalized tasks per run_id after activation.")
        else:
            st.info("There is no significant change after normalizing for business growth.")

    # ---------------------------
    # Percentage Change Calculation
    # ---------------------------
    st.subheader("Impact Analysis: Percentage Change in Normalized Tasks per Run ID")

    before_avg = df_filtered[df_filtered['year'] == (threshold.year - 1)]['normalized_avg_tasks_per_run'].mean()
    after_avg = df_filtered[df_filtered['year'] == threshold.year]['normalized_avg_tasks_per_run'].mean()

    if before_avg != 0:
        percent_change = ((after_avg - before_avg) / before_avg) * 100
        change_str = f"{abs(percent_change):.2f}% decrease" if percent_change < 0 else f"{percent_change:.2f}% increase"
        st.write(f"**Normalized Avg Tasks per Run:** Before = {before_avg:.4f}, After = {after_avg:.4f} ({change_str})")


    # ---------------------------
    # Optionally Show Raw Data
    # ---------------------------
    if st.checkbox("Show raw data"):
        st.subheader("Filtered Data")
        st.write(df_filtered)

else:
    st.info("Awaiting CSV file upload.")