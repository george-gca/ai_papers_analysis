import argparse
import logging
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import List

import comet_ml
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from paper_finder_trainer import PaperFinderTrainer
from timer import Timer
from utils import setup_log, supported_conferences


_logger = logging.getLogger(__name__)


def _add_abstract(row: pd.Series, unique_words: List[str]) -> None:
    words_list = row.clean_title.split()
    words_list += row.abstract.split()
    unique_words += list(set(words_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find word usage.")
    parser.add_argument('-c', '--conference', type=str, default='',
                        help='conference to scrape data')
    parser.add_argument('--clusters', type=int, default=26,
                        help='number of clusters to be divided')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory for the input data')
    parser.add_argument('--display_plot', action='store_true',
                        help='display clusters plot')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-m', '--model', type=str, default='skipgram',
                        choices=('skipgram', 'cbow'),
                        help='model trained for word representation')
    parser.add_argument('--model_dir', default='model_data',
                        type=str, help='directory for data')
    parser.add_argument('-n', '--max_ngram', type=int, default=5,
                        help='max n-gram of words to use')
    parser.add_argument('--word_dim', type=int, default=3,
                        help='dimensions for words representation')
    parser.add_argument('-x', '--perplexity', type=int,
                        default=25, help='perplexity param for t-SNE')
    parser.add_argument('-y', '--year', type=int, default=0,
                        help='year of the conference')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args.log_level, log_dir / 'cluster_conference_words.log')

    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    conferences = supported_conferences

    if len(args.conference) > 0:
        conferences = [ c for c in conferences if c.startswith(args.conference) ]

    if args.year > 0:
        conferences = [c for c in conferences if c.endswith(str(args.year))]

    # abstract_files = [data_dir / c / f'abstracts_{args.max_ngram}gram.csv' for c in conferences]
    abstract_files = [data_dir / c / 'abstracts_clean.csv' for c in conferences]

    _logger.print(f'Clustering words for {len(conferences)} conferences: {conferences}')

    p2v = PaperFinderTrainer(data_dir=data_dir, model_dir=model_dir)
    p2v.load_words_model(str(model_dir / f'fasttext_{args.model}_50000w.bin'))

    # comet ml logging
    experiment = comet_ml.Experiment(project_name='AI Papers', auto_metric_logging=False)
    experiment.set_name(f'Cluster Conference Words')
    experiment.log_parameters(args)

    for i, abstract_file in enumerate(abstract_files):
        abstract_unique_words = []
        df = pd.read_csv(abstract_file, sep='|', dtype=str, keep_default_na=False)
        # df = pd.read_feather(abstract_file)
        # df.dropna(inplace=True)

        experiment.log_dataframe_profile(df, minimal=False, log_raw_dataframe=True,
            name=f'{"_".join(conferences[i].split("/"))}_abstracts_{args.max_ngram}gram')

        tqdm.pandas(unit='paper', desc='Reading papers abstracts')
        df.progress_apply(_add_abstract, axis=1, unique_words=abstract_unique_words)

        unique_words_in_conference = set(abstract_unique_words)
        n_papers = len(df)
        n_words = len(unique_words_in_conference)
        _logger.print(
            f'{conferences[i]} has {n_words:n} unique words.')

        papers_w_words = Counter(abstract_unique_words).most_common()
        word_vector = np.zeros([n_words, p2v.word_dim])
        for j, word_count in enumerate(papers_w_words):
            word_vector[j] += p2v.model.get_word_vector(word_count[0])

        # log conference word vectors to comet ml
        name = f'{"_".join(conferences[i].split("/"))}_words'
        words_str = [['Word', 'Count']]
        words_str += [[w, c] for w, c in papers_w_words]
        experiment.log_embedding(word_vector, words_str,
                                 title=name,
                                 template_filename=name)

        _logger.print(
            f'Reducing word_vector from {word_vector.shape[1]} to {args.word_dim} dims')
        tsne = TSNE(perplexity=args.perplexity, n_components=args.word_dim,
                    init='pca', n_iter=2000, n_jobs=2*multiprocessing.cpu_count()//3)
        with Timer(name=f'Reducing dimensions'):
            word_vector = tsne.fit_transform(word_vector)

        _logger.print(f'Creating {args.clusters} clusters')
        estimator = KMeans(init='k-means++', n_clusters=args.clusters, n_init=10)
        estimator.fit(word_vector)
        cluster_ids = estimator.labels_

        # log conference word vectors clusterized to comet ml
        words_str = [['Word', 'Count', 'Cluster']]
        words_str += [[w, c, cl]
                      for (w, c), cl in zip(papers_w_words, cluster_ids)]
        experiment.log_embedding(word_vector, words_str,
                                 title=f'{name}_clusters',
                                 template_filename=f'{name}_clusters')

        print()

    experiment.log_asset(str(log_dir / 'cluster_conference_words.log'))
