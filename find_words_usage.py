import argparse
import csv
from itertools import islice, pairwise
import locale
import multiprocessing
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import comet_ml
import numpy as np
import pandas as pd
from prettytable import MARKDOWN, PrettyTable
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

from paper_finder_trainer import PaperFinderTrainer
from utils import recreate_url, setup_log, supported_conferences


_logger = logging.getLogger(__name__)

# Use '' for auto, or force e.g. to 'en_US.UTF-8'
locale.setlocale(locale.LC_ALL, '')


IGNORE_SET = {
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
        'learning',
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
        'state_of_the_art',
        'strong',
        'task',
        'training',
        'via',
        'way',
        'well',
        'without',
        'work',
    }


def _add_abstract(row: pd.Series, words: list[str], unique_words: list[str]) -> None:
    words_list = [w for w in row.clean_title.split() if len(w) > 1]
    words_list += [w for w in row.abstract.split() if len(w) > 1]
    words += words_list
    unique_words += list(set(words_list))


def _cluster_new_words(new_words_usage: list[tuple[str, int]], paper_finder: PaperFinderTrainer,
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

    n_dimensions = 5
    _logger.print(f'Reducing word_vector from {word_vector.shape[1]} to {n_dimensions} dims')
    tsne = TSNE(perplexity=25, n_components=n_dimensions, init='pca', n_iter=2000,
                n_jobs=2*multiprocessing.cpu_count()//3)
    word_vector = tsne.fit_transform(word_vector)

    n_clusters = 10
    _logger.print(f'Creating {n_clusters} clusters')
    estimator = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    estimator.fit(word_vector)
    cluster_ids = estimator.labels_

    # log conference word vectors clusterized to comet ml
    words_str = [['Word', 'Count', 'Cluster']]
    words_str += [[w, c, cl]
                  for (w, c), cl in zip(new_words_usage, cluster_ids)]
    experiment.log_embedding(word_vector, words_str,
                             title=f'{name}_clusters',
                             template_filename=f'{name}_clusters')


def _create_conferences_stats(conferences: list[str],
                              abstract_files: list[str]) -> \
                              tuple[list[dict[str, int]],
                                  list[dict[str, int]],
                                  list[set[str]],
                                  list[int]]:
    """
    Create statistics from each conference.

    Args:
        conferences: List of conferences.
        abstract_files: List of abstract files.

    Returns:
        occurrence_of_words_dict: List of dictionaries with the number of
            occurrences of each word in each conference.
        papers_with_word_dict: List of dictionaries with the number of papers
            that have each word in each conference.
        unique_words: List of sets with the unique words in each conference.
        n_papers: List of the number of papers in each conference.
    """

    occurrence_of_words_dict = []
    papers_with_word_dict = []
    unique_words = []
    n_papers = []

    for i, abstract_file in enumerate(abstract_files):
        abstract_words = []
        abstract_unique_words = []
        df = pd.read_csv(abstract_file, sep='|', dtype=str, keep_default_na=False)

        tqdm.pandas(unit='paper', desc='Reading papers abstracts')
        df.progress_apply(_add_abstract, axis=1, words=abstract_words,
                          unique_words=abstract_unique_words)

        unique_words_in_conference = set(abstract_unique_words)

        _logger.print(f'{conferences[i]} have {len(abstract_words):n} total words\n'
                      f'{len(abstract_unique_words):n} happens uniquely in each abstract.\n'
                      f'With a total of {len(unique_words_in_conference):n} unique words.\n')

        occurrences = Counter(abstract_words).most_common()
        papers_w_words = Counter(abstract_unique_words).most_common()

        occurrence_of_words_dict.append({k: v for k, v in occurrences})
        papers_with_word_dict.append({k: v for k, v in papers_w_words})
        unique_words.append(unique_words_in_conference)
        n_papers.append(len(df))

    # write this data to a file
    conference = conferences[0].split('/')[0]
    output_dir = Path('words_usage/')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f'{conference}_occurrences_of_words.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Count', 'Year'])

        for i, occurrences in enumerate(occurrence_of_words_dict):
            for word, count in occurrences.items():
                writer.writerow([word, count, conferences[i].split('/')[1]])

    with open(output_dir / f'{conference}_papers_with_word.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Count', 'Year'])

        for i, papers_w_words in enumerate(papers_with_word_dict):
            for word, count in papers_w_words.items():
                writer.writerow([word, count, conferences[i].split('/')[1]])

    with open(output_dir / f'{conference}_papers_per_year.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Count', 'Year'])

        for i, papers in enumerate(n_papers):
            writer.writerow([papers, conferences[i].split('/')[1]])

    return occurrence_of_words_dict, papers_with_word_dict, unique_words, n_papers


def _filter_and_cluster_papers(new_words_usage: list[tuple[str, int]], paper_finder: PaperFinderTrainer,
                               conference: str, year: int,
                               experiment: comet_ml.Experiment, args: argparse.Namespace) -> None:

    data_dir = Path(args.data_dir).expanduser()

    # filter new words that occurs less than 5 times
    keywords = tuple(w for w, c in new_words_usage if c >= 5)

    # find papers with new words on this conference/year
    results, _ = paper_finder.find_by_keywords(keywords, -1, similars=3, conference=conference, year=year)

    if len(results) == 0:
        _logger.print('No papers found.')
        return

    # cluster papers with new words
    papers_to_keep = {paper_finder.papers[r[0]].title for r in results}
    _logger.print(f'Keeping {len(papers_to_keep)} papers')

    n_keywords = 10

    # comet ml logging
    name = f'{conference}_{year}_papers_with_new_words'

    _logger.print('\nStep 1: Build paper representation vectors with fasttext.')
    paper_finder.build_paper_vectors(
        data_dir / f'abstracts_{args.max_ngram}gram.feather',
        suffix='_pwc',
        filter_titles=papers_to_keep)

    # log conference paper vectors to comet ml
    paper_titles = [['Title', 'Conference', 'Year', 'PDF']]
    paper_titles += [[t.title, t.conference, t.year, recreate_url(t.pdf_url, t.conference, t.year)]
                     for t in paper_finder.papers]
    experiment.log_embedding(paper_finder.paper_vectors, paper_titles, title=name, template_filename=name)

    _logger.print(
        '\nStep 2: Reduce dimensions and then apply k-means clustering.')
    paper_finder.reduce_paper_vectors_dim(3, 25)
    clusters = max(10, len(papers_to_keep) // 10)
    paper_finder.clustering_papers(clusters)

    # log conference paper vectors clusterized to comet ml
    paper_titles = [['Title', 'Conference', 'Year', 'Cluster', 'PDF']]
    paper_titles += [[t.title, t.conference, t.year, c, recreate_url(t.pdf_url, t.conference, t.year)]
                     for t, c in zip(paper_finder.papers, paper_finder.paper_cluster_ids)]
    experiment.log_embedding(paper_finder.paper_vectors, paper_titles,
                             title=f'clusters_{name}',
                             template_filename=f'clusters_{name}')

    for i in range(clusters):
        cluster_keywords = paper_finder.cluster_abstract_freq[i]
        cluster_keywords = list(islice((paper_finder.abstract_words[w] \
                                        for w, _ in cluster_keywords if w not in IGNORE_SET), n_keywords))
        _logger.print(f'cluster {i+1:02d} keywords: {", ".join(cluster_keywords)}')


def _print_most_used_new_words(
        new_words_usage: list[tuple[str, int]],
        paper_finder: PaperFinderTrainer,
        n_similar_words: int,
        conference: str,
        year: int,
        experiment = comet_ml.Experiment,
        ) -> None:
    new_words_usage = new_words_usage.copy()

    # write this data to a file
    output_dir = Path('words_usage/')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f'{conference}_{year}_new_words.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Count', 'Related words', 'Words weights'])

        for word, count in new_words_usage:
            similar_words = paper_finder.get_most_similar_words(word, n_similar_words)
            weights, words = zip(*similar_words)
            writer.writerow([word, count, ', '.join(words), ' '.join([str(w) for w in weights])])

    table = PrettyTable()
    table.field_names = ['Word', 'Related new word', '# Occurrences', 'Related words']

    i = 0
    max_words_per_line = 5

    while i < len(new_words_usage):
        word, count = new_words_usage[i]
        # discard new words with little usage
        if count < 5:
            break

        similar_words = paper_finder.get_most_similar_words(word, n_similar_words)
        similar_words = [w for _, w in similar_words]

        new_similar_words = []
        j = i+1
        while j < len(new_words_usage):
            # check for new words that are similar to the current new word
            if new_words_usage[j][0] in similar_words:
                new_similar_words.append(new_words_usage[j])
                similar_words.remove(new_words_usage[j][0])
                new_words_usage.remove(new_words_usage[j])
            else:
                j += 1

        if len(similar_words) <= max_words_per_line:
            similar_words = ', '.join(similar_words)
            table.add_row([word, '', count, similar_words])
        else:
            for k in range(0, len(similar_words), max_words_per_line):
                similar_words_group = ', '.join(similar_words[k:k+max_words_per_line])

                if k == max_words_per_line:
                    table.add_row([word, '', count, similar_words_group])
                else:
                    table.add_row(['', '', '', similar_words_group])

        if len(new_similar_words) > 0:
            for new_word, new_count in new_similar_words:
                similar_words = paper_finder.get_most_similar_words(new_word, n_similar_words)
                similar_words = [w for _, w in similar_words]

                if len(similar_words) <= max_words_per_line:
                    similar_words = ', '.join(similar_words)
                    table.add_row(['', new_word, new_count, similar_words])
                else:
                    for k in range(0, len(similar_words), max_words_per_line):
                        similar_words_group = ', '.join(similar_words[k:k+max_words_per_line])

                        if k == max_words_per_line:
                            table.add_row(['', new_word, new_count, similar_words_group])
                        else:
                            table.add_row(['', '', '', similar_words_group])

        i += 1

    table.set_style(MARKDOWN)
    _logger.print(f'\nMost used new words in {year}:\n\n{table}\n')
    experiment.log_table(f'{year} most used new words.csv', tabular_data=table.get_formatted_string('csv').replace('\r', ''), headers=True)


def _print_papers_with_words(new_words_usage: list[tuple[str, int]], paper_finder: PaperFinderTrainer,
                             conference: str, year: int) -> None:

    # filter new words that occurs less than 5 times
    keywords = [w for w, c in new_words_usage if c >= 5]

    _logger.print('\nFinding papers that uses the new words\n')
    not_found_keywords = set()

    # write this data to a file
    output_dir = Path('words_usage/')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f'{conference}_{year}_papers_with_new_words.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'Titles'])

        for keyword in keywords:
            results, _ = paper_finder.find_by_keywords(tuple(keyword.split()), -1, similars=0, conference=conference, year=year)

            if len(results) > 0:
                _logger.print(f'\nPapers that use the word: {keyword}')
                writer.writerow([keyword, '\n'.join([paper_finder.papers[paper_id].title for paper_id, _ in results])])

                for paper_id, _ in results:
                    _logger.print(f'\t{paper_finder.papers[paper_id].title}')

            else:
                not_found_keywords.add(keyword)

    _logger.print(f'\nNo papers found for words:\n{", ".join(sorted(not_found_keywords))}.')


def _sort_rows(rows: list[Any]) -> list[Any]:
    top_rows = [r for r in rows if r[2] == '↑']
    bottom_rows = [r for r in rows if r[2] == '↓']

    new_rows = sorted(top_rows, key=lambda x: x[0], reverse=True) + sorted(bottom_rows, key=lambda x: x[0])
    return [r[1:] for r in new_rows]


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

    assert len(args.conference) > 0, 'You must set a conference of search'

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    log_file = f'find_{args.conference}_words_usage.log'

    setup_log(args.log_level, log_dir / log_file)

    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    conferences = [c for c in supported_conferences if args.conference == c.split('/')[0]]

    abstract_files = [data_dir / c / f'abstracts_{args.max_ngram}gram.csv' for c in conferences]
    # abstract_files = [data_dir / c / 'abstracts_clean.csv' for c in conferences]

    # set up comet experiment
    experiment = comet_ml.Experiment(project_name='AI Papers', auto_metric_logging=False)
    experiment.set_name(f'New words in {args.conference}')
    experiment.log_parameters(args)

    # get data
    occurrence_of_words_dict, papers_with_word_dict, unique_words, n_papers = \
        _create_conferences_stats(conferences, abstract_files)

    # load embeddings model
    p2v = PaperFinderTrainer(data_dir=data_dir, model_dir=model_dir)
    p2v.load_words_model(str(model_dir / f'fasttext_{args.model}_50000w.bin'))
    p2v.load_paper_vectors(load_similar_dict=True, suffix=args.suffix)
    p2v.load_abstracts(data_dir / 'abstracts_pwc.feather')
    # p2v.load_abstracts(data_dir / 'abstracts_clean_pwc.feather')

    # create sequences of conferences (e.g. (cvpr/2019, cvpr/2020), (cvpr/2020, cvpr/2021))
    sequences = pairwise(range(len(conferences)))
    new_words = []
    # percentage of how many times the word was used to consider it a relevant change
    variation_in_all_words = 0.05
    # percentage of how many papers used the word to consider it a relevant change
    variation_of_word = 0.03
    n_similar_words = 7

    for c1, c2 in sequences:
        # get new words used in c2 that were not used in c1
        new_words_from = unique_words[c2] - unique_words[c2].intersection(unique_words[c1])
        _logger.print(f'\n{len(new_words_from):n} new words from {conferences[c1]} to {conferences[c2]}')
        new_words.append(new_words_from)

        conference, year = conferences[c2].split('/')
        year = int(year)

        # get the most used new words, most used first
        new_words_usage = {w: occurrence_of_words_dict[c2][w] for w in new_words_from}
        new_words_usage = [(k, v) for k, v in sorted(new_words_usage.items(), key=lambda item: item[1], reverse=True)]

        # TODO give the possibility to search which papers used the given word
        # _cluster_new_words(new_words_usage, p2v, conference, year, experiment)
        _print_most_used_new_words(new_words_usage, p2v, n_similar_words, conference, year, experiment)
        _print_papers_with_words(new_words_usage, p2v, conference, year)
        # _filter_and_cluster_papers(new_words_usage, p2v, conference, year, experiment, args)

        words_usage_decreased = []
        words_usage_increased = []
        same_words = {w for w in unique_words[c2].intersection(unique_words[c1]) if w not in IGNORE_SET}

        _logger.print(f'\nWords that had variation in amount of papers that use it (no matter how many times)'
                      f' bigger than {variation_of_word*100}%:\n')
        table = PrettyTable()
        table.field_names = [
            'Word',
            'Variation',
            f'# occurrences in {conferences[c1].split("/")[1]}', f'% occurrences in {conferences[c1].split("/")[1]}',
            f'# occurrences in {conferences[c2].split("/")[1]}', f'% occurrences in {conferences[c2].split("/")[1]}',
            ]
        rows = []

        for word in same_words:
            papers_in_c1 = papers_with_word_dict[c1][word] / n_papers[c1]
            papers_in_c2 = papers_with_word_dict[c2][word] / n_papers[c2]
            variation = abs(papers_in_c2 - papers_in_c1)

            if variation > variation_of_word:
                if papers_in_c2 > papers_in_c1:
                    words_usage_increased.append(word)
                    symbol = '↑'
                else:
                    words_usage_decreased.append(word)
                    symbol = '↓'

                rows.append([variation, word, symbol,
                               papers_with_word_dict[c1][word],
                               f'{papers_in_c1*100:.2f}',
                               papers_with_word_dict[c2][word],
                               f'{papers_in_c2*100:.2f}'])

        rows = _sort_rows(rows)
        table.add_rows(rows)
        table.set_style(MARKDOWN)
        _logger.print(f'\n{table}')
        experiment.log_table(
            f'{year} variation in # of papers using > {variation_of_word*100}%.csv',
            tabular_data=table.get_formatted_string('csv').replace('\r', ''),
            headers=True,
            )

        # print groups of words that increased papers using it
        _logger.print('\nGroup of words that increased papers using it:\n')
        i = 0

        while i < len(words_usage_increased):
            word = words_usage_increased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [w for w in words_usage_increased[i+1:] if w in similar_words]

            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(', '.join(similar_words_group))

                for w in similar_words_group[i+1:]:
                    words_usage_increased.remove(w)

            i += 1

        # print groups of words that decreased papers using it
        _logger.print('\nGroup of words that decreased papers using it:\n')
        i = 0

        while i < len(words_usage_decreased):
            word = words_usage_decreased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [w for w in words_usage_decreased[i+1:] if w in similar_words]

            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(', '.join(similar_words_group))

                for w in similar_words_group[i+1:]:
                    words_usage_decreased.remove(w)

            i += 1

        words_usage_decreased = []
        words_usage_increased = []

        _logger.print(f'\nWords that had variation in usage bigger than {variation_in_all_words*100}%:\n')
        table = PrettyTable()
        table.field_names = [
            'Word',
            'Variation',
            f'# occurrences in {conferences[c1].split("/")[1]}', f'% occurrences in {conferences[c1].split("/")[1]}',
            f'# occurrences in {conferences[c2].split("/")[1]}', f'% occurrences in {conferences[c2].split("/")[1]}',
            ]
        rows = []

        for word in same_words:
            words_in_c1 = occurrence_of_words_dict[c1][word] / n_papers[c1]
            words_in_c2 = occurrence_of_words_dict[c2][word] / n_papers[c2]
            variation = abs(words_in_c2 - words_in_c1)

            if variation > variation_in_all_words and word not in IGNORE_SET:
                if words_in_c2 > words_in_c1:
                    words_usage_increased.append(word)
                    symbol = '↑'
                else:
                    words_usage_decreased.append(word)
                    symbol = '↓'

                rows.append([variation, word, symbol,
                                 occurrence_of_words_dict[c1][word],
                                 f'{words_in_c1*100:.2f}',
                                 occurrence_of_words_dict[c2][word],
                                 f'{words_in_c2*100:.2f}'])

        rows = _sort_rows(rows)
        table.add_rows(rows)
        table.set_style(MARKDOWN)
        _logger.print(f'\n{table}')
        experiment.log_table(
            f'{year} variation in usage > {variation_of_word*100}%.csv',
            tabular_data=table.get_formatted_string('csv').replace('\r', ''),
            headers=True,
            )

        # print groups of words that usage increased
        _logger.print('\nGroup of words that usage increased:\n')
        i = 0

        while i < len(words_usage_increased):
            word = words_usage_increased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [w for w in words_usage_increased[i+1:] if w in similar_words]

            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(', '.join(similar_words_group))

                for w in similar_words_group[i+1:]:
                    words_usage_increased.remove(w)

            i += 1

        # print groups of words that usage decreased
        _logger.print('\nGroup of words that usage decreased:\n')
        i = 0

        while i < len(words_usage_decreased):
            word = words_usage_decreased[i]
            similar_words = p2v.get_most_similar_words(word, 10)
            similar_words = [w for _, w in similar_words]
            similar_words_group = [w for w in words_usage_decreased[i+1:] if w in similar_words]

            if len(similar_words_group) > 0:
                similar_words_group = [word] + similar_words_group
                _logger.print(', '.join(similar_words_group))

                for w in similar_words_group[i+1:]:
                    words_usage_decreased.remove(w)

            i += 1

    experiment.log_asset(str(log_dir / log_file))
