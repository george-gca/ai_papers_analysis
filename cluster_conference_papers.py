import argparse
import datetime
import logging
from pathlib import Path

import comet_ml
from paper_finder_trainer import PaperFinderTrainer
from utils import NOT_INFORMATIVE_WORDS, recreate_url_from_code, setup_log, SUPPORTED_CONFERENCES


_logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conference', type=str, default='',
                        help='conference to scrape data')
    parser.add_argument('--clusters', type=int, default=18,
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
    parser.add_argument('-p', '--paper_dim', type=int, default=3,
                        help='dimensions for paper representation')
    parser.add_argument('-x', '--perplexity', type=int,
                        default=25, help='perplexity param for t-SNE')
    parser.add_argument('-y', '--year', type=int, default=0,
                        help='year of the conference')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args.log_level, log_dir / 'cluster_conference_papers.log')

    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    conferences = [f'{c}/{y}' for c in SUPPORTED_CONFERENCES for y in range(2017, datetime.date.today().year + 1)]
    conferences = [c for c in conferences if (data_dir / c).exists()]

    if len(args.conference) > 0:
        conferences = [c for c in conferences if c.startswith(args.conference)]

    if args.year > 0:
        conferences = [c for c in conferences if c.endswith(str(args.year))]

    # abstract_files = (data_dir / c / f'abstracts_{args.max_ngram}gram.csv' for c in conferences)
    abstract_files = (data_dir / c / 'abstracts_clean.csv' for c in conferences)
    abstract_files = [c for c in abstract_files if c.exists()]

    _logger.print(f'Clustering papers for {len(conferences)} conferences: {conferences}')

    p2v = PaperFinderTrainer(data_dir=data_dir, model_dir=model_dir)
    p2v.load_words_model(str(model_dir / f'fasttext_{args.model}_50000w.bin'))

    n_keywords = 15

    for i, abstract_file in enumerate(abstract_files):
        # comet ml logging
        experiment = comet_ml.Experiment(project_name='AI Papers', auto_metric_logging=False)
        experiment.set_name(f'Cluster {" ".join(conferences[i].split("/"))} papers')
        experiment.log_parameters(args)

        _logger.print(
            f'\nStep 1: Build paper representation vectors for {conferences[i]} with fasttext.')
        p2v.build_paper_vectors(abstract_file)

        # log conference paper vectors to comet ml
        conference, year = conferences[i].split('/')
        name = f'{conference}_{year}_papers'
        year = int(year)
        paper_titles = [['Title', 'Conference', 'Year', 'PDF']]
        paper_titles += [[t.title, conference, year, recreate_url_from_code(t.pdf_url, t.source_url, conference, year)]
                        for t in p2v.papers]
        experiment.log_embedding(p2v.paper_vectors, paper_titles,
                                 title=name,
                                 template_filename=name)

        _logger.print(
            '\nStep 2: Reduce dimensions and then apply k-means clustering.')
        p2v.reduce_paper_vectors_dim(args.paper_dim, perplexity=args.perplexity)
        p2v.clustering_papers(clusters=args.clusters)

        # log conference paper vectors clusterized to comet ml
        paper_titles = [['Title', 'Conference', 'Year', 'Cluster', 'PDF']]
        paper_titles += [[t.title, conference, year, c, recreate_url_from_code(t.pdf_url, t.source_url, conference, year)]
                        for t, c in zip(p2v.papers, p2v.paper_cluster_ids)]
        experiment.log_embedding(p2v.paper_vectors, paper_titles,
                                 title=f'{name}_clusters',
                                 template_filename=f'{name}_clusters')

        for i in range(args.clusters):
            cluster_keywords = p2v.cluster_abstract_freq[i]
            cluster_keywords = [
                p2v.abstract_words[w] for w, _ in cluster_keywords if w not in NOT_INFORMATIVE_WORDS][:n_keywords]
            _logger.print(
                f'cluster {i+1:02d} keywords: {", ".join(cluster_keywords)}')

        experiment.log_asset(str(log_dir / 'cluster_conference_papers.log'))
        experiment.end()
