import argparse
import locale
import multiprocessing
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import comet_ml
import numpy as np
import pandas as pd
from colorama import Fore
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from paper_finder_trainer import PaperFinderTrainer
from utils import recreate_url, setup_log, supported_conferences


_logger = logging.getLogger(__name__)

# Use '' for auto, or force e.g. to 'en_US.UTF-8'
locale.setlocale(locale.LC_ALL, '')


def _add_abstract(row: pd.Series, words: List[str], unique_words: List[str]) -> None:
    words_list = row.clean_title.split()
    words_list += row.abstract.split()
    words += words_list
    unique_words += list(set(words_list))


def _cluster_new_words(new_words_usage: List[Tuple[str, int]], paper_finder: PaperFinderTrainer,
                       conference: str, year: int, experiment: comet_ml.Experiment) -> None:
    word_vector = np.zeros([len(new_words_usage), paper_finder.word_dim])
    for i, word_count in enumerate(new_words_usage):
        word_vector[i] += paper_finder.model.get_word_vector(word_count[0])

    # log conference word vectors to comet ml
    name = f'{conference}_{year}_new_words'
    words_str = [['Word', 'Count']]
    words_str += [[w, c] for w, c in new_words_usage]
    experiment.log_embedding(word_vector, words_str,
                             title=name,
                             template_filename=name)

    _logger.print(f'Reducing word_vector from {word_vector.shape[1]} to 3 dims')
    tsne = TSNE(perplexity=25, n_components=3, init='pca', n_iter=2000,
                n_jobs=2*multiprocessing.cpu_count()//3)
    word_vector = tsne.fit_transform(word_vector)

    _logger.print(f'Creating 50 clusters')
    estimator = KMeans(init='k-means++', n_clusters=50, n_init=10)
    estimator.fit(word_vector)
    cluster_ids = estimator.labels_

    # log conference word vectors clusterized to comet ml
    words_str = [['Word', 'Count', 'Cluster']]
    words_str += [[w, c, cl]
                  for (w, c), cl in zip(new_words_usage, cluster_ids)]
    experiment.log_embedding(word_vector, words_str,
                             title=f'{name}_clusters',
                             template_filename=f'{name}_clusters')


def _create_sequences_of_conferences(conferences: List[str]) -> List[str]:
    return [l for l in zip(*(conferences[i:] for i in range(2)))]


def _filter_and_cluster_papers(new_words_usage: List[Tuple[str, int]], conference: str, year: int,
                               experiment: comet_ml.Experiment, args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()

    p2v = PaperFinderTrainer(data_dir=data_dir, model_dir=model_dir)
    p2v.load_paper_vectors(load_similar_dict=True, suffix=args.suffix)
    p2v.load_abstracts(data_dir / 'abstracts_pwc.feather')
    p2v.load_words_model(str(model_dir / f'fasttext_{args.model}_50000w.bin'))

    # filter new words that occurs less than 6 times
    keywords = tuple([w for w, c in new_words_usage if c > 5])

    results, _ = p2v.find_by_keywords(
        keywords, -1, similars=5, conference=conference, year=year)

    if len(results) <= 0:
        _logger.print('No papers found.')
        return

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
    name = f'{conference}_{year}_papers_with_new_words'

    _logger.print(f'\nStep 1: Build paper representation vectors with fasttext.')
    p2v.build_paper_vectors(f'abstracts_{args.max_ngram}gram.feather', filter_titles=papers_to_keep)

    # log conference paper vectors to comet ml
    paper_titles = [['Title', 'Conference', 'Year', 'PDF']]
    paper_titles += [[t.title, t.conference, t.year, recreate_url(t.pdf_url, t.conference, t.year)]
                     for t in p2v.papers]
    experiment.log_embedding(
        p2v.paper_vectors, paper_titles, title=name, template_filename=name)

    _logger.print(
        '\nStep 2: Reduce dimensions and then apply k-means clustering.')
    p2v.reduce_paper_vectors_dim(3, 25)
    clusters = 10
    p2v.clustering_papers(clusters)

    # log conference paper vectors clusterized to comet ml
    paper_titles = [['Title', 'Conference', 'Year', 'Cluster', 'PDF']]
    paper_titles += [[t.title, t.conference, t.year, c, recreate_url(t.pdf_url, t.conference, t.year)]
                     for t, c in zip(p2v.papers, p2v.paper_cluster_ids)]
    experiment.log_embedding(p2v.paper_vectors, paper_titles,
                             title=f'clusters_{name}',
                             template_filename=f'clusters_{name}')

    for i in range(clusters):
        cluster_keywords = p2v.cluster_abstract_freq[i]
        cluster_keywords = [
            p2v.abstract_words[w] for w, _ in cluster_keywords if w not in not_informative_words][:n_keywords]
        _logger.print(
            f'cluster {i+1:02d} keywords: {", ".join(cluster_keywords)}')


def _print_most_used_new_words(new_words_usage: List[Tuple[str, int]], paper_finder: PaperFinderTrainer,
                               n_similar_words: int, conference: str, year: int,
                               experiment: comet_ml.Experiment) -> None:
    _cluster_new_words(new_words_usage, paper_finder, conference, year, experiment)

    word_count_len = max([len(w) for w, _ in new_words_usage])
    most_used_new_words = []

    i = 0
    while i < len(new_words_usage):
        word, count = new_words_usage[i]
        if count <= 5:
            break

        similar_words = paper_finder.get_most_similar_words(
            word, n_similar_words)
        similar_words = [w for _, w in similar_words]

        new_similar_words = []
        j = i+1
        while j < len(new_words_usage):
            if new_words_usage[j][0] in similar_words:
                new_similar_words.append(new_words_usage[j])
                similar_words.remove(new_words_usage[j][0])
                new_words_usage.remove(new_words_usage[j])
            else:
                j += 1

        similar_words = ', '.join(similar_words)
        most_used_new_words.append(f'{word:<{word_count_len}}: {count:02} - Related words: {similar_words}')

        if len(new_similar_words) > 0:
            for w, c in new_similar_words:
                most_used_new_words.append(
                    f'{w:<{word_count_len}}: {c:02} - Related to {word}')

        i += 1

    most_used_new_words = '\n\t'.join(most_used_new_words)
    _logger.print(
        f'\nMost used new words:\n\t{most_used_new_words}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Find word usage.")
    parser.add_argument('-c', '--conference', type=str, default='cvpr',
                        help='conference to scrape data')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory for the input data')
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
    parser.add_argument('--suffix', default='_70_clusters',
                        type=str, help='suffix of model files to use')
    # parser.add_argument('-u', '--unique', action='store_true',
    #                     help='consider unique words per paper, not count all occurrences of the word')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args.log_level, log_dir / 'find_words_usage.log')

    ignore_set = {
        'accuracy',
        'achieve',
        'aim',
        'algorithm',
        'available',
        'better',
        'case',
        'data',
        'deep',
        'demonstrate',
        'directly',
        'effective',
        'either',
        'evaluation',
        'even',
        'existing',
        'goal',
        'make',
        'may',
        'method',
        'model',
        'module',
        'need',
        'network',
        'new',
        'novel',
        'performance',
        'present',
        'previous',
        'prior',
        'problem',
        'result',
        'set',
        'several',
        'show',
        'simple',
        'state-of-the-art',
        'strong',
        'training',
        'via',
        'way',
        'well',
        'without',
        'work',
    }

    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    conferences = supported_conferences

    if len(args.conference) > 0:
        conferences = [c for c in conferences if args.conference in c]

    # abstract_files = [data_dir / c / f'abstracts_{args.max_ngram}gram.csv' for c in conferences]
    abstract_files = [data_dir / c / 'abstracts_clean.csv' for c in conferences]

    occurence_of_words = []
    occurence_of_words_dict = []
    papers_with_word = []
    papers_with_word_dict = []
    unique_words = []
    n_papers = []

    experiment = comet_ml.Experiment(project_name='AI Papers', auto_metric_logging=False)
    experiment.set_name(f'New words in {args.conference}')
    experiment.log_parameters(args)

    for i, abstract_file in enumerate(abstract_files):
        abstract_words = []
        abstract_unique_words = []
        df = pd.read_csv(
            abstract_file, sep='|', dtype=str, keep_default_na=False)

        tqdm.pandas(unit='paper', desc='Reading papers abstracts')
        df.progress_apply(_add_abstract, axis=1, words=abstract_words,
                          unique_words=abstract_unique_words)

        unique_words_in_conference = set(abstract_unique_words)

        _logger.print(f'{conferences[i]} have {len(abstract_words):n} total words\n'
                      f'{len(abstract_unique_words):n} happens uniquely in each abstract.\n'
                      f'With a total of {len(unique_words_in_conference):n} unique words.\n')

        occurrences = Counter(abstract_words).most_common()
        papers_w_words = Counter(abstract_unique_words).most_common()

        occurence_of_words.append(occurrences)
        papers_with_word.append(papers_w_words)
        occurence_of_words_dict.append({k: v for k, v in occurrences})
        papers_with_word_dict.append({k: v for k, v in papers_w_words})
        unique_words.append(unique_words_in_conference)
        n_papers.append(len(df))

    p2v = PaperFinderTrainer(data_dir=data_dir, model_dir=model_dir)
    p2v.load_words_model(str(model_dir / f'fasttext_{args.model}_50000w.bin'))
    p2v.load_paper_vectors(load_similar_dict=True, suffix=args.suffix)
    p2v.load_abstracts(data_dir / 'abstracts_pwc.feather')
    # p2v.load_abstracts(data_dir / 'abstracts_clean_pwc.feather')

    sequences = _create_sequences_of_conferences(list(range(len(conferences))))
    new_words = []
    variation_in_all_words = 0.05
    variation_of_word = 0.03
    n_similar_words = 7

    for c1, c2 in sequences:
        new_words_from = unique_words[c2] - \
            unique_words[c2].intersection(unique_words[c1])
        _logger.print(
            f'\n{len(new_words_from):n} new words from {conferences[c1]} to {conferences[c2]}')
        new_words.append(new_words_from)

        conference, year = conferences[c2].split('/')
        year = int(year)

        new_words_usage = {
            w: occurence_of_words_dict[c2][w] for w in new_words_from}
        new_words_usage = [(k, v) for k, v in sorted(
            new_words_usage.items(), key=lambda item: item[1], reverse=True)]

        # TODO from most used new words, get similar words with fasttext and check which words occurred together, create clusters
        # to give a better notion of new themes, with phrases rather than just words
        # TODO give the possibility to search which papers used the given word
        # TODO group new words by context. e.g.: deepglobe and land are both from agriculture
        _print_most_used_new_words(new_words_usage, p2v, n_similar_words, conference, year, experiment)
        _filter_and_cluster_papers(new_words_usage, conference, year, experiment, args)

        variations_text = []
        words_usage_decreased = []
        words_usage_increased = []
        same_words = unique_words[c2].intersection(unique_words[c1])

        for word in same_words:
            papers_in_c1 = papers_with_word_dict[c1][word] / n_papers[c1]
            papers_in_c2 = papers_with_word_dict[c2][word] / n_papers[c2]
            if abs(papers_in_c2 - papers_in_c1) > variation_of_word and word not in ignore_set:
                if papers_in_c2 > papers_in_c1:
                    variation = f'Papers with {word} {Fore.GREEN} ↑{variation_of_word*100}%:'
                    words_usage_increased.append(word)
                else:
                    variation = f'Papers with {word} {Fore.YELLOW} ↓{variation_of_word*100}%:'
                    words_usage_decreased.append(word)

                variation += f' {papers_with_word_dict[c1][word]:n}({papers_in_c1*100:.2f}%) → ' \
                             f'{papers_with_word_dict[c2][word]:n}({papers_in_c2*100:.2f}%).{Fore.RESET}'

                variations_text.append(variation)

        for text in sorted(variations_text):
            _logger.print(text)

        _logger.print('')
        i = 0
        while i < len(words_usage_increased):
            word = words_usage_increased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [
                w for w in words_usage_increased[i+1:] if w in similar_words]
            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(
                    f'Group of words that increased papers using it: {similar_words_group}')
                for w in similar_words_group[i+1:]:
                    words_usage_increased.remove(w)
            i += 1

        _logger.print('')
        i = 0
        while i < len(words_usage_decreased):
            word = words_usage_decreased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [
                w for w in words_usage_decreased[i+1:] if w in similar_words]
            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(
                    f'Group of words that decreased papers using it: {similar_words_group}')
            i += 1

        _logger.print('')
        variations_text = []
        words_usage_decreased = []
        words_usage_increased = []
        for word in same_words:
            words_in_c1 = occurence_of_words_dict[c1][word] / n_papers[c1]
            words_in_c2 = occurence_of_words_dict[c2][word] / n_papers[c2]
            if abs(words_in_c2 - words_in_c1) > variation_in_all_words and word not in ignore_set:
                if words_in_c2 > words_in_c1:
                    variation = f'{word} occurrence {Fore.GREEN} ↑ {variation_in_all_words*100}%:'
                    words_usage_increased.append(word)
                else:
                    variation = f'{word} occurrence {Fore.YELLOW} ↓ {variation_in_all_words*100}%:'
                    words_usage_decreased.append(word)

                variation += f' {occurence_of_words_dict[c1][word]:n}({words_in_c1*100:.2f}%) → ' \
                             f'{occurence_of_words_dict[c2][word]:n}({words_in_c2*100:.2f}%).{Fore.RESET}'

                variations_text.append(variation)

        for text in sorted(variations_text):
            _logger.print(text)

        _logger.print('')
        i = 0
        while i < len(words_usage_increased):
            word = words_usage_increased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [
                w for w in words_usage_increased[i+1:] if w in similar_words]
            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(
                    f'Group of words that increased usage: {similar_words_group}')
            i += 1

        _logger.print('')
        i = 0
        while i < len(words_usage_decreased):
            word = words_usage_decreased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [
                w for w in words_usage_decreased[i+1:] if w in similar_words]
            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(
                    f'Group of words that decreased usage: {similar_words_group}')
            i += 1

    experiment.log_asset(str(log_dir / 'find_words_usage.log'))
