import os
import streamlit as st

st.set_page_config(
    page_title="CodeGauge",
    page_icon="<Target>",
    layout="wide",
)

conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))

st.title("CodeGauge")
st.caption("Predict DSA problem difficulty using ML")

st.divider()

col_input, col_info = st.columns([2, 1])

with col_input:
    search_mode = st.radio(
        "Search by",
        ["Problem Name", "Contest ID"],
        horizontal=True,
    )

    if search_mode == "Problem Name":
        user_input = st.text_input(
            "Enter a problem name",
            placeholder="e.g. Binary Search, Watermelon, Two Sum",
        )
    else:
        user_input = st.text_input(
            "Enter a contest ID",
            placeholder="e.g. 1945, 1436, 100",
        )

with col_info:
    st.markdown(
        """
        **How it works**
        1. Type a problem name or contest ID
        2. The ML model predicts Easy / Medium / Hard
        3. We cross-check against the actual difficulty
        """
    )

if user_input:
    safe_input = user_input.strip().replace("'", "''")

    if search_mode == "Contest ID":
        if not safe_input.isdigit():
            st.warning("Please enter a numeric contest ID.")
            st.stop()
        where_clause = f"s.CONTESTID = {safe_input}"
    else:
        where_clause = f"s.NAME ILIKE '%{safe_input}%'"

    query = f"""
    SELECT
        s.NAME AS PROBLEM_NAME,
        s.CONTESTID,
        s.INDEX,
        s.TOTAL_SUBMISSIONS,
        s.UNIQUE_USERS,
        s.ACCEPTANCE_RATE,
        s.DIFFICULTY AS ACTUAL_DIFFICULTY,
        CODEGAUGE_DB.ML.CODEGAUGE_CLASSIFIER!PREDICT(
            INPUT_DATA => OBJECT_CONSTRUCT(
                'CONTESTID', s.CONTESTID,
                'INDEX', s.INDEX,
                'TAGS', s.TAGS,
                'TOTAL_SUBMISSIONS', s.TOTAL_SUBMISSIONS,
                'ACCEPTED_SUBMISSIONS', s.ACCEPTED_SUBMISSIONS,
                'WRONG_ANSWERS', s.WRONG_ANSWERS,
                'UNIQUE_USERS', s.UNIQUE_USERS,
                'ACCEPTANCE_RATE', s.ACCEPTANCE_RATE,
                'WRONG_RATE', ROUND(s.WRONG_ANSWERS / NULLIF(s.TOTAL_SUBMISSIONS, 0), 6),
                'USER_SUBMISSION_RATIO', ROUND(s.UNIQUE_USERS / NULLIF(s.TOTAL_SUBMISSIONS, 0), 6),
                'OTHER_VERDICTS', s.TOTAL_SUBMISSIONS - s.ACCEPTED_SUBMISSIONS - s.WRONG_ANSWERS,
                'TAG_COUNT', ARRAY_SIZE(SPLIT(s.TAGS, ',')),
                'WRONG_PER_USER', ROUND(s.WRONG_ANSWERS / NULLIF(s.UNIQUE_USERS, 0), 6),
                'SUBMISSIONS_PER_USER', ROUND(s.TOTAL_SUBMISSIONS / NULLIF(s.UNIQUE_USERS, 0), 6)
            )
        ):class::VARCHAR AS PREDICTED_DIFFICULTY
    FROM CODEGAUGE_DB.ML.ML_SPLIT s
    WHERE {where_clause}
    ORDER BY s.TOTAL_SUBMISSIONS DESC
    LIMIT 20
    """

    with st.spinner("Running prediction..."):
        df = conn.query(query)

    if df.empty:
        st.info(f"No problems found matching **{user_input}**. Try a different search.")
    else:
        df["VERDICT"] = df.apply(
            lambda r: "Correct" if r["ACTUAL_DIFFICULTY"] == r["PREDICTED_DIFFICULTY"] else "Wrong",
            axis=1,
        )
        correct = (df["VERDICT"] == "Correct").sum()
        total = len(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Problems Found", total)
        m2.metric("Correct Predictions", correct)
        m3.metric("Accuracy", f"{correct / total:.0%}")

        st.divider()

        for _, row in df.iterrows():
            is_correct = row["VERDICT"] == "Correct"
            icon = "white_check_mark" if is_correct else "x"
            color = "green" if is_correct else "red"

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.markdown(f"**{row['PROBLEM_NAME']}** (Contest {row['CONTESTID']}{row['INDEX']})")
                c2.markdown(f"Predicted: **:{color}[{row['PREDICTED_DIFFICULTY']}]**")
                c3.markdown(f"Actual: **{row['ACTUAL_DIFFICULTY']}**")
                c4.markdown(f":{icon}: **{row['VERDICT']}**")

                with st.expander("Details"):
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Total Submissions", f"{row['TOTAL_SUBMISSIONS']:,}")
                    d2.metric("Unique Users", f"{row['UNIQUE_USERS']:,}")
                    d3.metric("Acceptance Rate", f"{row['ACCEPTANCE_RATE']:.1%}")

    st.divider()

st.markdown(
    """
    ---
    **About CodeGauge** | Model: `SNOWFLAKE.ML.CLASSIFICATION` trained on 14.9M Codeforces submissions
    | Accuracy: ~82% on held-out test set | Features: 14 (behavioral + contextual, no rating leakage)
    """
)
