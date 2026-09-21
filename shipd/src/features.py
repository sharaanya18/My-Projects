"""Feature construction. Every block is switchable so ablations are honest."""
from __future__ import annotations
import numpy as np, pandas as pd
from data import CAT_COLS, num_cols

# Features whose train->test drift is largest (KS / TVD from the Phase 6 audit).
SHIFTED = ['body_mention_count', 'body_mention_band', 'body_heading_count',
           'body_line_count', 'title_colon_count', 'other_file_count',
           'body_code_block_count', 'body_checklist_item_count', 'body_structure']

# Features that look like project-template identifiers rather than mechanisms.
FINGERPRINT = ['title_digit_count', 'body_digit_count', 'body_line_count',
               'title_colon_count', 'body_unique_character_count',
               'title_length', 'body_length', 'body_unique_word_count']

# Causally plausible drivers of review volume (should transfer across projects).
CAUSAL = ['test_file_count', 'docs_file_count', 'code_file_count', 'config_file_count',
          'other_file_count', 'file_count', 'additions', 'deletions', 'changed_lines',
          'max_file_changes', 'max_file_additions', 'max_file_deletions',
          'test_term', 'docs_term', 'security_term', 'bug_term', 'backport_term',
          'release_term', 'base_channel', 'change_scope', 'body_structure',
          'body_code_block_count', 'body_question_count', 'body_word_count']


def cat_dtypes(TR, TE):
    ALL = pd.concat([TR.drop(columns=[c for c in ['target'] if c in TR]), TE], ignore_index=True)
    return {c: pd.CategoricalDtype(sorted(ALL[c].astype(str).unique())) for c in CAT_COLS}


def derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and densities: scale-free, so they survive a size shift better
    than raw counts do."""
    e = 1e-6
    d = pd.DataFrame(index=df.index)
    d['test_file_ratio']   = df.test_file_count / (df.file_count + e)
    d['docs_file_ratio']   = df.docs_file_count / (df.file_count + e)
    d['code_file_ratio']   = df.code_file_count / (df.file_count + e)
    d['config_file_ratio'] = df.config_file_count / (df.file_count + e)
    d['other_file_ratio']  = df.other_file_count / (df.file_count + e)
    d['add_del_ratio']     = df.additions / (df.deletions + e)
    d['churn_per_file']    = df.changed_lines / (df.file_count + e)
    d['max_file_share']    = df.max_file_changes / (df.changed_lines + e)
    d['del_share']         = df.deletions / (df.changed_lines + e)
    d['body_per_line']     = df.body_length / (df.body_line_count + 1)
    d['body_word_density'] = df.body_unique_word_count / (df.body_word_count + e)
    d['title_word_density']= df.title_unique_word_count / (df.title_word_count + e)
    d['body_digit_rate']   = df.body_digit_count / (df.body_length + e)
    d['title_digit_rate']  = df.title_digit_count / (df.title_length + e)
    d['body_ref_rate']     = df.body_reference_count / (df.body_word_count + e)
    d['body_link_rate']    = df.body_link_count / (df.body_word_count + e)
    d['has_body']          = (df.body_length > 0).astype(float)
    d['has_tests']         = (df.test_file_count > 0).astype(float)
    d['has_docs']          = (df.docs_file_count > 0).astype(float)
    d['log_changed']       = np.log1p(df.changed_lines)
    d['log_files']         = np.log1p(df.file_count)
    d['log_body']          = np.log1p(df.body_length)
    return d


def build(TR, TE, *, use_raw=True, use_cat=True, use_derived=True,
          drop=(), keep_only=None, dtypes=None):
    """Assemble the design matrices. `drop`/`keep_only` drive the ablations."""
    dtypes = dtypes or cat_dtypes(TR, TE)
    num = [c for c in num_cols(TR)]

    def one(df):
        parts = []
        if use_raw:
            parts.append(df[num].astype(float))
        if use_derived:
            parts.append(derived(df))
        X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
        if use_cat:
            for c in CAT_COLS:
                X[c] = df[c].astype(str).astype(dtypes[c])
        return X

    A, B = one(TR), one(TE)
    if keep_only is not None:
        cols = [c for c in A.columns if c in set(keep_only)]
        A, B = A[cols], B[cols]
    if drop:
        dropset = set(drop)
        cols = [c for c in A.columns if c not in dropset]
        A, B = A[cols], B[cols]
    return A, B
