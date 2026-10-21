/* Worked example challenges the platform publishes for solvers. Each one shows
 * every part of a challenge (dataset description, prepare script, problem
 * description, grading script, config, rubric, reference solution and score),
 * condensed here into what each part tells you and where the reference leaves
 * room. Scores are the ones reported on the example page; the improvement
 * ideas are hypotheses until you validate them. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  E.EXAMPLES = [
    {
      slug: "notebook-upvotes",
      title: "Notebook Upvote Prediction",
      domain: "Tabular regression",
      task: "tabular",
      category: "general",
      metric: "maelog",
      dir: "min",
      difficulty: "medium",
      summary: "Predict a Kaggle notebook's upvote count from metadata: author stats, code metrics, topics, libraries, dates and the title. 56,762 training rows, 6,971 test rows. Mixed feature types are the point of the challenge.",
      reference: { score: 1.061, note: "MAE in log space, as reported on the example page (the page also shows 1.06)." },
      anatomy: [
        ["Dataset description", "Every column and its type, which columns are entirely null, the date format (dd/mm/yyyy), the pipe-separated columns, and that the target is right-skewed (median 1, mean 26, max 10,469). Your preprocessing plan, written for you."],
        ["Prepare script", "Keeps notebook records only, then train_test_split(test_size=0.1, random_state=0) by row. The hidden test is a random slice, so shuffled KFold mirrors it and the same authors appear in train and test."],
        ["Problem description", "The metric, MAE on log(y+1) capped at 999, and why: the log stops the long tail dominating. That says what to predict (a median in log space), not just what to report."],
        ["Grading script", "Columns must be exactly {content_id, upvotes}; same row count as the answers; every value ≥ 0, or it raises; both files are sorted by content_id, so row order doesn't matter. There is no NaN check."],
        ["Config", "direction: minimize, minimum 0, maximum 999. Lower is better, and a broken file can't score better than the cap."],
        ["Rubric", "11 specific, checkable criteria: 4 REQUIRED, 7 RECOMMENDED, in Data Handling, Feature Engineering, Modeling and Agent Behavior. This is what a reviewer looks for in your code."],
        ["Reference solution", "A complete, readable script and its score. It's the floor to beat, and a recipe you can re-run on your own folds to see how CV maps onto the leaderboard."],
      ],
      rubric: [
        ["required", "Data Handling", "Identify and remove target-leaking columns (fork_count, views, downloads, comments_count, notebook_usage, medal, is_featured, is_trending, engagement_rate, virality_score, quality_score): future information not available at creation time."],
        ["required", "Data Handling", "Log-transform the upvote count to handle the right-skewed distribution and match the evaluation metric."],
        ["required", "Data Handling", "Split all_topics and libraries_used on the pipe delimiter and encode the multi-label values (one-hot or ordinal)."],
        ["recommended", "Agent Behavior", "Remove features that are entirely null or irrelevant (usability_score, file_format, column_count, row_count, license_type)."],
        ["recommended", "Data Handling", "Convert created_date and last_updated from dd/mm/yyyy into numbers that keep temporal order."],
        ["recommended", "Data Handling", "Handle the title with TF-IDF, embeddings or similar NLP preprocessing."],
        ["recommended", "Modeling", "Use a gradient boosting model (e.g. XGBoost, LightGBM) for this regression."],
        ["recommended", "Feature Engineering", "Create interaction terms (e.g. days_since_creation / update_count for update frequency)."],
        ["required", "Modeling", "No significant overfitting: the gap between training and validation performance is reasonable."],
        ["recommended", "Modeling", "Run a hyperparameter search (grid search or Optuna)."],
        ["recommended", "Modeling", "Clip final predictions so upvote counts are non-negative."],
      ],
      recipe: [
        "Drops the leakage list (only the columns that are present) and the entirely-null columns",
        "One 80/20 train_test_split for validation (random_state=42)",
        "StandardScaler on numerics, OneHotEncoder on categoricals, dates to days since 1970",
        "MultiLabelBinarizer on all_topics and libraries_used, TF-IDF (256 terms) on the title, ordinal-encoded author_username",
        "RandomForestRegressor (100 trees) on log1p(upvotes), validation MAE printed in log space",
        "expm1, clip at 0, write content_id + upvotes",
      ],
      room: [
        "MSE trees fit the mean of log1p(y); the metric rewards the median. Try an L1 or quantile-0.5 objective on log1p(y).",
        "One holdout split, and the final model never sees the 20% validation rows. Use K-fold OOF and refit on all rows with a fixed iteration count.",
        "No train-fit score is printed, yet one REQUIRED criterion is about the train/validation gap. Print both.",
        "Authors recur between train and test (random split). Out-of-fold author statistics (mean log-upvotes, count) may carry signal; ordinal codes don't.",
        "Dates are parsed without format=. Pass format=\"%d/%m/%Y\" so day and month can't swap.",
        "Two RECOMMENDED items are skipped: gradient boosting and interaction terms. Hyperparameter search too.",
        "The CSV isn't validated before writing. Assert the grader's own checks: exact columns, row count, ≥ 0, finite.",
      ],
      lessons: ["Read the prepare script before choosing CV", "Treat the rubric as part of the spec", "The grader's own checks decide errors", "Encode each feature type on purpose", "Beat a reference solution with metric-matched training"],
      // condensed text for the analyzer: the parts it reads, in the page's own order
      description: `# Notebook Upvote Prediction
Predict Kaggle notebook upvote counts from metadata including author stats, code metrics, and text features. A tabular regression problem with diverse feature types.

## Dataset description
62,733 notebook records (train.csv 56,762 rows with labels; test.csv 6,971 rows without the upvotes column; sample_submission.csv with random values). Several features that could leak the target variable (e.g. number of comments, number of forks) were removed.
Columns: content_id, title (free text), author_username, author_tier (Novice / Contributor / Expert / Master / Grandmaster), author_followers, author_notebooks_count, author_datasets_count, primary_topic (50 categories), all_topics (pipe-separated topic list), programming_language, is_competition_related, created_date (dd/mm/yyyy), last_updated (dd/mm/yyyy), days_since_creation, update_count, execution_time_seconds, code_lines, markdown_ratio, visualization_count, libraries_used (pipe-separated library list), uses_gpu, file_size_mb.
- Missing values: usability_score, file_format, column_count, row_count, license_type are entirely null (dataset-only fields).
- Target distribution: right-skewed. Median 1, mean 26, max 10,469.
- The title column is free text requiring NLP preprocessing.

## Prepare script
data = data[data.content_type == "notebook"]
train, test = train_test_split(data, test_size=0.1, random_state=0)

## Evaluation
Submissions are evaluated using a modified mean absolute error (MAE) in log space:
MAE = min(mean(|log(y_i + 1) - log(ŷ_i + 1)|), 999)
The log transformation handles the right-skewed distribution.

## Submission
| Column | Type |
|--------|------|
| content_id | string |
| upvotes | float |

## Grading script
def grade(submission, answers):
    if set(submission.columns) != {"content_id", "upvotes"}: raise Exception("columns")
    if len(submission) != len(answers): raise Exception("rows")
    if not (submission['upvotes'] >= 0).all(): raise Exception("non-negative")
    submission_sorted = submission.sort_values(by="content_id")
    return min(mae_log_score, 999)

## Config
difficulty: medium
domain: tabular
grade: direction: minimize, minimum: 0, maximum: 999

## Rubric
`,
    },
  ];

  // the analyzer reads the rubric in the page's own badge / category / criterion layout
  E.EXAMPLES.forEach((x) => {
    x.description += x.rubric.map(([lvl, cat, text]) => `${lvl.toUpperCase()}\n${cat}\n${text}`).join("\n") +
      `\n\n## Reference solution\nRandom forest on log-transformed targets, achieving a MAE of ${x.reference.score} in log space.\n`;
  });
})();
