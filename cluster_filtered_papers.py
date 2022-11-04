import argparse
import locale
import logging
from pathlib import Path
from typing import List

import comet_ml

from paper_finder_trainer import PaperFinderTrainer
from utils import recreate_url, setup_log


_logger = logging.getLogger(__name__)

# Use '' for auto, or force e.g. to 'en_US.UTF-8'
locale.setlocale(locale.LC_ALL, '')


def define_keywords(keywords_text: str) -> List[str]:
    keywords = keywords_text.strip().lower()
    if '"' in keywords:
        while '"' in keywords:
            first_index = keywords.find('"')
            last_index = keywords[first_index+1:].find('"')

            if last_index == -1:
                keywords = keywords.replace('"', ' ')
                break

            joined_words = keywords[first_index+1:first_index+last_index+1]
            joined_words = '_'.join(joined_words.split())
            keywords = f'{keywords[:first_index]}' \
                f'{keywords[first_index+1:first_index+last_index+1]}' \
                f'{keywords[first_index+last_index+2:]} {joined_words}'

    return keywords.split()


def filter_and_cluster_papers(args: argparse.Namespace):
    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args.log_level, log_dir / 'clusterize_filtered_papers.log')

    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()

    p2v = PaperFinderTrainer(data_dir=data_dir, model_dir=model_dir)
    p2v.load_paper_vectors(load_similar_dict=True, suffix=args.suffix)
    p2v.load_abstracts(data_dir / 'abstracts_pwc.feather')
    # p2v.load_abstracts(data_dir / 'abstracts_clean_pwc.feather')
    p2v.load_words_model(str(model_dir / f'fasttext_{args.model}_50000w.bin'))

    keywords = define_keywords(args.keywords)

    _logger.print(f'\nKeyword(s): {keywords}')
    exclude_keywords = []
    for k in keywords:
        if k.startswith('-'):
            keywords.remove(k)
            exclude_keywords.append(k[1:])

    results, _ = p2v.find_by_keywords(
        tuple(keywords), args.count, conference=args.conference, year=args.year, exclude_keywords=tuple(exclude_keywords), search_str=args.keywords)

    if len(results) <= 0:
        _logger.print('No papers found.')
        exit(0)

    papers_to_keep = {p2v.papers[r[0]].title for r in results}
    _logger.print(f'Keeping {len(papers_to_keep)} papers')

    n_keywords = 15
    not_informative_words = [
        'data',
        'learning',
        'method',
        'model',
        'network',
        'problem',
        'result',
        'task',
        'training'
    ]

    # comet ml logging
    if len(args.name) == 0:
        name = ''
        if len(keywords) > 0:
            name += f'{"_".join(keywords)}'
        if len(exclude_keywords) > 0:
            name += f'_{"_-".join(exclude_keywords)}'
        if len(args.conference) > 0:
            name += f'_{args.conference}'
        if args.year > 0:
            name += f'_{args.year}'
    else:
        name = args.name

    experiment = comet_ml.Experiment(
        project_name='AI Papers', auto_metric_logging=False)
    experiment.set_name(name)
    experiment.log_parameters(args)

    _logger.print(
        f'\nStep 1: Build paper representation vectors with fasttext.')
    p2v.build_paper_vectors(data_dir / f'abstracts_{args.max_ngram}gram.feather', suffix='_pwc', filter_titles=papers_to_keep)
    # p2v.build_paper_vectors(data_dir / 'abstracts_clean_pwc.feather', filter_titles=papers_to_keep)


    # log conference paper vectors to comet ml
    paper_titles = [['Title', 'Conference', 'Year', 'PDF']]
    paper_titles += [[t.title, t.conference, t.year, recreate_url(t.pdf_url, t.conference, t.year)]
                     for t in p2v.papers]
    experiment.log_embedding(
        p2v.paper_vectors, paper_titles, title=name, template_filename=name)

    _logger.print(
        '\nStep 2: Reduce dimensions and then apply k-means clustering.')
    p2v.reduce_paper_vectors_dim(
        args.paper_dim, perplexity=args.perplexity)
    p2v.clustering_papers(clusters=args.clusters)

    # log conference paper vectors clusterized to comet ml
    paper_titles = [['Title', 'Conference', 'Year', 'Cluster', 'PDF']]
    paper_titles += [[t.title, t.conference, t.year, c, recreate_url(t.pdf_url, t.conference, t.year)]
                     for t, c in zip(p2v.papers, p2v.paper_cluster_ids)]
    experiment.log_embedding(p2v.paper_vectors, paper_titles,
                             title=f'clusters_{name}',
                             template_filename=f'clusters_{name}')

    for i in range(args.clusters):
        cluster_keywords = p2v.cluster_abstract_freq[i]
        cluster_keywords = [
            p2v.abstract_words[w] for w, _ in cluster_keywords if w not in not_informative_words][:n_keywords]
        _logger.print(
            f'cluster {i+1:02d} keywords: {", ".join(cluster_keywords)}')

    experiment.log_asset(str(log_dir / 'clusterize_filtered_papers.log'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('keywords', type=str, help='keywords for search')
    parser.add_argument('-c', '--conference', type=str, default='',
                        help='conference to scrape data')
    parser.add_argument('--clusters', type=int, default=26,
                        help='number of clusters to be divided')
    parser.add_argument('--count', default=-1, type=int,
                        help='max number of papers to find')
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
    parser.add_argument('--name', type=str, default='',
                        help='name to be used for experiment')
    parser.add_argument('-n', '--max_ngram', type=int, default=5,
                        help='max n-gram of words to use')
    parser.add_argument('-p', '--paper_dim', type=int, default=3,
                        help='dimensions for paper representation')
    parser.add_argument('--suffix', default='_70_clusters',
                        type=str, help='suffix of model files to use')
    parser.add_argument('-x', '--perplexity', type=int,
                        default=25, help='perplexity param for t-SNE')
    parser.add_argument('-y', '--year', type=int, default=0,
                        help='year of the conference')
    args = parser.parse_args()

    filter_and_cluster_papers(args)
