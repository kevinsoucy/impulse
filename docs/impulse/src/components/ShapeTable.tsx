import React from 'react';
import clsx from 'clsx';

type Col = {
    /** column header label */
    label: string;
    /** true for engine-role columns (session / signal / time / entity), tinted distinctly from payload */
    role?: boolean;
};

type Highlight = {
    /** zero-based row indices that share one instant */
    rows: number[];
    /** zero-based column index whose value is shared across the highlighted rows */
    col: number;
    /** annotation shown beneath the table */
    note: string;
};

type ShapeTableProps = {
    title: string;
    badge: '1.0' | '2.0';
    caption: string;
    columns: Col[];
    rows: (string | number)[][];
    highlight?: Highlight;
    className?: string;
};

/**
 * Renders one measurement-data "shape" as a styled mini-table: role columns
 * (session / signal / time / entity) are tinted apart from payload columns, and
 * an optional highlight bands a group of rows that share a single timestamp —
 * the multiplicity that scalar channels cannot represent.
 */
export default function ShapeTable({
                                       title,
                                       badge,
                                       caption,
                                       columns,
                                       rows,
                                       highlight,
                                       className,
                                   }: ShapeTableProps) {
    const isV2 = badge === '2.0';
    const highlightRows = new Set(highlight?.rows ?? []);

    return (
        <div
            className={clsx(
                'rounded-xl border overflow-hidden flex flex-col',
                'border-[var(--ifm-color-emphasis-200)] bg-[var(--ifm-background-surface-color)]',
                className,
            )}
        >
            {/* header */}
            <div className="px-4 py-3 border-b border-[var(--ifm-color-emphasis-200)] flex items-start gap-3">
                <span
                    className={clsx(
                        'font-mono text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 mt-0.5',
                        isV2
                            ? 'bg-indigo-500 text-white'
                            : 'bg-[var(--ifm-color-emphasis-200)] text-[var(--ifm-color-emphasis-700)]',
                    )}
                >
                    {badge}
                </span>
                <div>
                    <div className="font-sans font-semibold leading-tight">{title}</div>
                    <div className="text-sm text-[var(--ifm-color-emphasis-600)] leading-snug">{caption}</div>
                </div>
            </div>

            {/* grid "table" — divs, not <table>, to avoid global table CSS */}
            <div
                className="grid font-mono text-xs overflow-x-auto"
                style={{gridTemplateColumns: `repeat(${columns.length}, minmax(max-content, 1fr))`}}
            >
                {/* header row */}
                {columns.map((c, ci) => (
                    <div
                        key={`h-${ci}`}
                        className={clsx(
                            'px-3 py-2 border-b border-[var(--ifm-color-emphasis-200)] font-semibold whitespace-nowrap',
                            c.role
                                ? 'bg-[var(--ifm-color-emphasis-100)] text-[var(--ifm-color-emphasis-800)]'
                                : 'text-[var(--ifm-color-emphasis-700)]',
                        )}
                    >
                        {c.label}
                        {c.role && (
                            <span className="ml-1 text-[10px] uppercase tracking-wide text-[var(--ifm-color-emphasis-500)]">
                                role
                            </span>
                        )}
                    </div>
                ))}

                {/* data rows */}
                {rows.map((row, ri) => {
                    const inGroup = highlightRows.has(ri);
                    const firstOfGroup = inGroup && !highlightRows.has(ri - 1);
                    return row.map((cell, ci) => {
                        const isHighlightCol = highlight && inGroup && ci === highlight.col;
                        return (
                            <div
                                key={`r-${ri}-${ci}`}
                                className={clsx(
                                    'px-3 py-1.5 whitespace-nowrap border-b border-[var(--ifm-color-emphasis-100)]',
                                    columns[ci]?.role && 'bg-[var(--ifm-color-emphasis-100)]',
                                    inGroup && 'bg-amber-50 dark:bg-amber-400/10',
                                    isHighlightCol && 'bg-amber-100 dark:bg-amber-400/25 font-semibold',
                                    inGroup && ci === 0 && 'border-l-4 border-l-amber-400',
                                    firstOfGroup && 'border-t-2 border-t-amber-400',
                                )}
                            >
                                {cell}
                            </div>
                        );
                    });
                })}
            </div>

            {/* footnote / highlight annotation */}
            {highlight && (
                <div className="px-4 py-2.5 text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-400/10 border-t border-[var(--ifm-color-emphasis-200)] flex items-start gap-2">
                    <span aria-hidden className="font-bold">↳</span>
                    <span>{highlight.note}</span>
                </div>
            )}
        </div>
    );
}
