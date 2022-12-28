import argparse
import logging
from pathlib import Path
from multiprocessing import cpu_count

from top2vec import Top2Vec

from utils import create_corpus, setup_log


_logger = logging.getLogger(__name__)


def _train_top2vec_model(speed: str, conference: str, year: int) -> None:
    if len(conference) > 0 and year > 0:
        conf_year = f'_{conference}_{year}'
    elif len(conference) > 0:
        conf_year = f'_{conference}'
    elif year > 0:
        conf_year = f'_{year}'
    else:
        conf_year = ''

    all_titles = Path(f'data/papers_titles{conf_year}.txt')
    all_texts = Path(f'data/papers_contents{conf_year}.txt')

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

    model.save(f'model_data/top2vec_model_{speed}{conf_year}')


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
        conf_year = f'_{args.conference}_{args.year}'
    elif len(args.conference) > 0:
        conf_year = f'_{args.conference}'
    elif args.year > 0:
        conf_year = f'_{args.year}'
    else:
        conf_year = ''

    setup_log(args.log_level, log_dir / f'top2vec{conf_year}.log')

    if args.create_corpus:
        create_corpus(args.separator, args.conference, args.year)

    if args.train:
        _train_top2vec_model(args.speed, args.conference, args.year)

    model = Top2Vec.load(f'model_data/top2vec_model_{args.speed}{conf_year}')

    if len(args.conference) > 0 and args.year > 0:
        _logger.print(f'Found {model.get_num_topics()} topics for {args.conference} {args.year}')
    elif len(args.conference) > 0:
        _logger.print(f'Found {model.get_num_topics()} topics for {args.conference}')
    elif args.year > 0:
        _logger.print(f'Found {model.get_num_topics()} topics for {args.year}')
    else:
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