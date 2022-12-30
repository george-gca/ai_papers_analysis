import argparse
import logging
from pathlib import Path

from tqdm import tqdm
from Doc2Map import Doc2Map

from utils import create_corpus, setup_log


_logger = logging.getLogger(__name__)


def _train_doc2map_model(speed: str, conference: str, year: int) -> None:
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
    all_urls = Path(f'data/papers_urls{conf_year}.txt')

    all_titles = all_titles.read_text().strip().split('\n')
    all_texts = all_texts.read_text().strip().split('\n')
    all_urls = all_urls.read_text().strip().split('\n')

    _logger.print(f'Found {len(all_titles):n} titles')
    _logger.print(f'Found {len(all_texts):n} papers')
    _logger.print(f'Found {len(all_urls):n} papers with urls')

    d2m = Doc2Map(speed=speed, lemmatizing=False)

    for p, t, u in zip(tqdm(all_texts, ncols=150), all_titles, all_urls):
        d2m.add_text(p, t, url=u)

    # Compute the data
    d2m.build()

    # Generate the interactive map with automatic zoom (the first example)
    out_dir = f'doc2map/{conf_year[1:]}/'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d2m.interactive_map(suffix=conf_year, out_dir=out_dir, display=False)
    d2m.interactive_tree(suffix=conf_year, out_dir=out_dir, display=False)

    # You can also generate tree, to visualize in 3D the linkage of your documents
    # It will produce the visuals see in the story telling article about Doc2Map
    # d2m.display_tree(suffix=conf_year, out_dir=out_dir, display=False)
    # d2m.display_simplified_tree(suffix=conf_year, out_dir=out_dir, display=False)
    # d2m.scatter(suffix=conf_year, out_dir=out_dir, display=False)

    # Generate the interactive map with manual zoom (the second example)
    # d2m.plotly_interactive_map(suffix=conf_year, out_dir=out_dir, display=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train doc2map model.")
    parser.add_argument('--conference', type=str, default='',
                        help='use papers from this conference only when creating corpus')
    parser.add_argument('-c', '--create_corpus', action='store_true',
                        help='create single corpus with entire text from all papers')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-s', '--separator', type=str, default='|',
                        help='csv separator')
    parser.add_argument('--speed', type=str, default='learn',
                        choices=['fast-learn', 'learn', 'deep-learn'])
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

    _train_doc2map_model(args.speed, args.conference, args.year)
