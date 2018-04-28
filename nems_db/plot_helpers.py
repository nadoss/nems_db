import logging

import pandas.io.sql as psql

import nems_db.plots as plots
from nems_db.db import Session, NarfResults, NarfBatches, get_batch_cells

log = logging.getLogger(__name__)


def plot_filtered_batch(batch, models, measure, plot_type,
                        only_fair=True, include_outliers=False, display=True,
                        extra_cols=[], snr=0.0, iso=0.0, snr_idx=0.0):
    cells = get_batch_cells(batch)['cellid'].tolist()
    cells = get_filtered_cells(cells, snr, iso, snr_idx)
    plot = get_plot(cells, models, measure, plot_type, only_fair,
                    include_outliers, display)
    plot.generate_plot()
    return plot


def get_plot(cells, models, measure, plot_type, only_fair=True,
             include_outliers=False, display=True):
    session = Session()
    results_df = psql.read_sql_query(
            session.query(NarfResults)
            .filter(NarfResults.cellid.in_(cells))
            .filter(NarfResults.modelname.in_(models))
            .statement, session.bind
            )
    results_models = [
            m for m in
            list(set(results_df['modelname'].values.tolist()))
            ]
    ordered_models = [
            m for m in models
            if m in results_models
            ]
    PlotClass = getattr(plots, plot_type)
    plot = PlotClass(
            data=results_df, measure=measure, models=ordered_models,
            fair=only_fair, outliers=include_outliers, display=display
            )
    session.close()
    return plot


def get_filtered_cells(cells, batch, snr=0.0, iso=0.0, snr_idx=0.0):
    """Removes cellids from list if they do not meet snr/iso criteria."""
    session = Session()
    snr = max(snr, 0)
    iso = max(iso, 0)
    snr_idx = max(snr_idx, 0)

    db_criteria = psql.read_sql_query(
            session.query(NarfBatches)
            .filter(NarfBatches.cellid.in_(cells))
            .statement, session.bind
            )

    if db_criteria.empty:
        log.warning("No matching cells found in NarfBatches,"
                    " no cellids were filtered.")
        return cells
    else:
        def filter_cells(cell):
            min_snr = min(cell.est_snr, cell.val_snr)
            min_isolation = cell.min_isolation
            min_snr_idx = cell.min_snr_index

            a = (snr > min_snr)
            b = (iso > min_isolation)
            c = (snr_idx > min_snr_idx)

            if a or b or c:
                try:
                    cells.remove(cell.cellid)
                except ValueError:
                    # cell already removed - necessary b/c pandas
                    # tries function twice on first row, which causes
                    # an error since our applied function has side-effects.
                    pass
            return

        db_criteria.apply(filter_cells, axis=1)

    return cells
