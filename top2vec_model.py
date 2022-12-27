import argparse
import logging
from pathlib import Path
from multiprocessing import cpu_count

import pandas as pd
from top2vec import Top2Vec
from tqdm import tqdm

from utils import conferences_pdfs, recreate_url, setup_log


_logger = logging.getLogger(__name__)


def _create_corpus(separator: str, conference: str, year: int) -> None:
    if len(conference) > 0 and year > 0:
        corpus_files = [Path(f'data/{c}/pdfs_clean.csv') for c in conferences_pdfs if c == f'{conference}/{year}']
        url_files = [Path(f'data/{c}/paper_info.csv') for c in conferences_pdfs if c == f'{conference}/{year}']

        all_titles = Path(f'data/{conference}_{year}_papers_titles.txt').open('w')
        all_texts = Path(f'data/{conference}_{year}_papers_contents.txt').open('w')
        all_urls = Path(f'data/{conference}_{year}_papers_urls.txt').open('w')
    elif len(conference) > 0:
        corpus_files = [Path(f'data/{c}/pdfs_clean.csv') for c in conferences_pdfs if c.startswith(conference)]
        url_files = [Path(f'data/{c}/paper_info.csv') for c in conferences_pdfs if c.startswith(conference)]

        all_titles = Path(f'data/{conference}_papers_titles.txt').open('w')
        all_texts = Path(f'data/{conference}_papers_contents.txt').open('w')
        all_urls = Path(f'data/{conference}_papers_urls.txt').open('w')
    elif year > 0:
        corpus_files = [Path(f'data/{c}/pdfs_clean.csv') for c in conferences_pdfs if c.endswith(str(year))]
        url_files = [Path(f'data/{c}/paper_info.csv') for c in conferences_pdfs if c.endswith(str(year))]

        all_titles = Path(f'data/{year}_papers_titles.txt').open('w')
        all_texts = Path(f'data/{year}_papers_contents.txt').open('w')
        all_urls = Path(f'data/{year}_papers_urls.txt').open('w')
    else:
        corpus_files = [Path(f'data/{c}/pdfs_clean.csv') for c in conferences_pdfs]
        url_files = [Path(f'data/{c}/paper_info.csv') for c in conferences_pdfs]

        all_titles = Path(f'data/papers_titles.txt').open('w')
        all_texts = Path(f'data/papers_contents.txt').open('w')
        all_urls = Path(f'data/papers_urls.txt').open('w')


    pbar_files = tqdm(corpus_files)
    titles_set = set()

    for i, (corpus_file, url_file) in enumerate(zip(pbar_files, url_files)):
        pbar_files.set_description(str(corpus_file.parents[0]).replace(str(corpus_file.parents[2]), '')[1:])
        if len(separator) == 1:
            df = pd.read_csv(corpus_file, sep=separator, dtype=str, keep_default_na=False)
        else:
            df = pd.read_csv(corpus_file, sep=separator, dtype=str, engine='python', keep_default_na=False)

        df_url = pd.read_csv(url_file, sep=';', dtype=str, keep_default_na=False)
        if len(df) < len(df_url):
                # drop extra urls
            papers_titles = set(df['title'])
            df_url = df_url[df_url['title'].isin(papers_titles)]

        assert len(df) == len(df_url), f'df ({len(df)}) and df_url ({len(df_url)}) should have same size'
        df = df.join(df_url['abstract_url'].astype(str))

        for title, text, url in zip(tqdm(df['title'], leave=False), df['paper'], df['abstract_url']):
            if title.lower() in titles_set:
                continue

            titles_set.add(title.lower())
            all_titles.write(f'{title}\n')
            all_texts.write(f'{text}\n')
            conf, year = conferences_pdfs[i].split('/')
            all_urls.write(f'{recreate_url(str(url), conf, int(year), is_abstract=True)}\n')

        all_titles.flush()
        all_texts.flush()
        all_urls.flush()

    all_titles.close()
    all_texts.close()
    all_urls.close()


def _train_top2vec_model(speed: str, conference: str, year: int) -> None:
    if len(conference) > 0 and year > 0:
        all_titles = Path(f'data/{conference}_{year}_papers_titles.txt')
        all_texts = Path(f'data/{conference}_{year}_papers_contents.txt')
    elif len(conference) > 0:
        all_titles = Path(f'data/{conference}_papers_titles.txt')
        all_texts = Path(f'data/{conference}_papers_contents.txt')
    elif year > 0:
        all_titles = Path(f'data/{year}_papers_titles.txt')
        all_texts = Path(f'data/{year}_papers_contents.txt')
    else:
        all_titles = Path(f'data/papers_titles.txt')
        all_texts = Path(f'data/papers_contents.txt')

    all_titles = all_titles.read_text().strip().split('\n')
    all_texts = all_texts.read_text().strip().split('\n')

    _logger.print(f'Found {len(all_titles):n} titles')
    _logger.print(f'Found {len(all_texts):n} papers')

    model = Top2Vec(
            all_texts,
            ngram_vocab=True,
            split_documents=True,
            use_corpus_file=True,
            document_ids=all_titles,
            keep_documents=False,
            speed=speed,
            workers=cpu_count()//2,
        )

    if len(conference) > 0 and year > 0:
        model.save(f'model_data/top2vec_model_{speed}_{conference}_{year}')
    elif len(conference) > 0:
        model.save(f'model_data/top2vec_model_{speed}_{conference}')
    elif year > 0:
        model.save(f'model_data/top2vec_model_{speed}_{year}')
    else:
        model.save(f'model_data/top2vec_model_{speed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train top2vec model.")
    parser.add_argument('--conference', type=str, default='',
                        help='use papers from this conference only when creating corpus')
    parser.add_argument('-c', '--create_corpus', action='store_true',
                        help='create single corpus with entire text from all papers')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-n', '--n_topics', type=int, default=5,
                        help='number of topics to search for')
    parser.add_argument('-s', '--separator', type=str, default='|',
                        help='csv separator')
    parser.add_argument('--speed', type=str, default='learn',
                        choices=['fast-learn', 'learn', 'deep-learn'])
    parser.add_argument('--search', type=str, nargs='*', default=[],
                        help='search for topics related to each of the given words')
    parser.add_argument('-t', '--train', action='store_true',
                        help='train top2vec model')
    parser.add_argument('--year', type=int, default=0,
                        help='use papers from this year only when creating corpus')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)

    if len(args.conference) > 0 and args.year > 0:
        setup_log(args.log_level, log_dir / f'top2vec_{args.conference}_{args.year}.log')
    elif len(args.conference) > 0:
        setup_log(args.log_level, log_dir / f'top2vec_{args.conference}.log')
    elif args.year > 0:
        setup_log(args.log_level, log_dir / f'top2vec_{args.year}.log')
    else:
        setup_log(args.log_level, log_dir / 'top2vec.log')

    if args.create_corpus:
        _create_corpus(args.separator, args.conference, args.year)

    if args.train:
        _train_top2vec_model(args.speed, args.conference, args.year)

    if len(args.conference) > 0 and args.year > 0:
        model = Top2Vec.load(f'model_data/top2vec_model_{args.speed}_{args.conference}_{args.year}')
        _logger.print(f'Found {model.get_num_topics()} topics for {args.conference} {args.year}')
    elif len(args.conference) > 0:
        model = Top2Vec.load(f'model_data/top2vec_model_{args.speed}_{args.conference}')
        _logger.print(f'Found {model.get_num_topics()} topics for {args.conference}')
    elif args.year > 0:
        model = Top2Vec.load(f'model_data/top2vec_model_{args.speed}_{args.year}')
        _logger.print(f'Found {model.get_num_topics()} topics for {args.year}')
    else:
        model = Top2Vec.load(f'model_data/top2vec_model_{args.speed}')
        _logger.print(f'Found {model.get_num_topics()} topics')

    # printing information about model
    topic_sizes, _ = model.get_topic_sizes()
    topic_words, word_scores, topic_nums = model.get_topics()

    for topic_num, topic_size, words, scores in zip(topic_nums, topic_sizes, topic_words, word_scores):
        _logger.print(f'\nTopic {topic_num} has {topic_size} documents')
        topic_word_scores = [f'{score:.3f} - {word}' for score, word in zip(scores, words)]
        topic_word_scores_str = '\n\t'.join(topic_word_scores)
        _logger.print(f'Most important words:\n\t{topic_word_scores_str}')

    for keyword in args.search:
        _logger.print(f'\nSearching for {args.n_topics} topics related to "{keyword}"')
        try:
            topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=[keyword], num_topics=args.n_topics)
            for topic_num, topic_score, words, scores in zip(topic_nums, topic_scores, topic_words, word_scores):
                _logger.print(f'\nTopic {topic_num} has score {topic_score:.3f}')
                topic_word_scores = [f'{score:.3f} - {word}' for score, word in zip(scores, words)]
                topic_word_scores_str = '\n\t'.join(topic_word_scores)
                _logger.print(f'Most similar words:\n\t{topic_word_scores_str}')
        except ValueError:
            _logger.print(f'\n"{keyword}" has not been learned by the model so it cannot be searched')