import argparse
import logging
from pathlib import Path
from multiprocessing import cpu_count

import pandas as pd
from top2vec import Top2Vec
from tqdm import tqdm

from utils import conferences_pdfs, recreate_url, setup_log


_logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train top2vec model.")
    parser.add_argument('-c', '--create_corpus', action='store_true',
                        help='create single corpus with entire text from all papers')
    parser.add_argument('--clean', action='store_true',
                        help='create single corpus with cleaned text from all papers')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('--speed', type=str, default='learn',
                        choices=['fast-learn', 'learn', 'deep-learn'])
    parser.add_argument('-s', '--separator', type=str, default='|',
                        help='csv separator')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args.log_level, log_dir / 'train_top2vec.log')

    if args.create_corpus:
        corpus_files = [Path(f'data/{c}/pdfs_clean.csv') for c in conferences_pdfs]
        url_files = [Path(f'data/{c}/paper_info.csv') for c in conferences_pdfs]

        all_titles = Path(f'data/papers_titles.txt').open('w')
        all_texts = Path(f'data/papers_contents.txt').open('w')
        all_urls = Path(f'data/papers_urls.txt').open('w')

        pbar_files = tqdm(corpus_files)
        titles_set = set()

        for i, (corpus_file, url_file) in enumerate(zip(pbar_files, url_files)):
            pbar_files.set_description(str(corpus_file.parents[0]).replace(str(corpus_file.parents[2]), '')[1:])
            if len(args.separator) == 1:
                df = pd.read_csv(corpus_file, sep=args.separator, dtype=str, keep_default_na=False)
            else:
                df = pd.read_csv(corpus_file, sep=args.separator, dtype=str, engine='python', keep_default_na=False)

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

    all_titles = Path(f'data/papers_titles.txt').read_text().strip().split('\n')
    all_texts = Path(f'data/papers_contents.txt').read_text().strip().split('\n')

    _logger.print(f'Found {len(all_titles):n} titles')
    _logger.print(f'Found {len(all_texts):n} papers')

    model = Top2Vec(
        all_texts,
        ngram_vocab=True,
        split_documents=True,
        use_corpus_file=True,
        document_ids=all_titles,
        keep_documents=False,
        speed=args.speed,
        workers=cpu_count()//2,
    )

    model.save(f'model_data/top2vec_model_{args.speed}')
