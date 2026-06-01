import React from 'react';
import {Database, Filter, GitMerge, Target, ArrowDown} from 'lucide-react';

type Source = {
    table: string;
    kind: string;
    predicate: string;
};

const SOURCES: Source[] = [
    {table: 'traffic_signs', kind: 'series', predicate: 'value == "30"'},
    {table: 'Vehicle Speed', kind: 'channel', predicate: '> 30'},
    {table: 'lidar_detections', kind: 'series', predicate: 'class == "cyclist" & dist < 8m'},
];

/** A faint vertical down-arrow used between flow stages, with an optional label. */
function Down({label}: {label?: string}) {
    return (
        <div className="flex flex-col items-center py-1">
            {label && (
                <span className="font-mono text-[11px] font-medium text-[var(--ifm-color-emphasis-600)]">
                    {label}
                </span>
            )}
            <ArrowDown className="w-5 h-5 text-[var(--ifm-color-emphasis-400)]" aria-hidden/>
        </div>
    );
}

/** Left-rail stage label (MAP / REDUCE). */
function StageLabel({text}: {text: string}) {
    return (
        <div className="hidden md:flex items-center">
            <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[var(--ifm-color-emphasis-500)] -rotate-90 whitespace-nowrap">
                {text}
            </span>
        </div>
    );
}

/**
 * Visualises a TSAL cross-series query as a map-reduce: a metadata pre-filter
 * narrows recordings, each table applies its own predicate locally in one pass
 * (the map), then TSAL combines the per-source intervals (the reduce).
 */
export default function QueryFlow() {
    const cell = 'rounded-lg border border-[var(--ifm-color-emphasis-200)] bg-[var(--ifm-background-surface-color)]';

    return (
        <div className="not-prose my-6 font-sans">
            {/* Stage 1 — metadata pre-filter */}
            <div className={clsxJoin(cell, 'px-4 py-3 flex items-center gap-3 max-w-md mx-auto')}>
                <Database className="w-5 h-5 text-[var(--ifm-color-primary)] shrink-0"/>
                <div>
                    <div className="font-semibold text-sm">container_metrics — metadata filter</div>
                    <div className="text-xs text-[var(--ifm-color-emphasis-600)]">
                        brand, date range, campaign → <strong>candidate recordings</strong>
                    </div>
                </div>
            </div>

            <Down/>

            {/* Stage 2 — MAP: each table filters locally, one pass per recording */}
            <div className="grid grid-cols-[auto_1fr] gap-2">
                <StageLabel text="map"/>
                <div className="rounded-xl border border-dashed border-indigo-300 dark:border-indigo-500/50 bg-indigo-50/50 dark:bg-indigo-500/5 p-3">
                    <div className="flex items-center gap-2 mb-1 text-indigo-700 dark:text-indigo-300">
                        <Filter className="w-4 h-4"/>
                        <span className="text-xs font-semibold uppercase tracking-wide">
                            Filter at each table
                        </span>
                    </div>
                    <div className="text-xs text-[var(--ifm-color-emphasis-600)] mb-3">
                        Any number of tables — <strong>no limit</strong>. The engine still resolves them in
                        <strong> one pass per recording</strong> (table count adds no stages).
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                        {SOURCES.map((s) => (
                            <div key={s.table} className={clsxJoin(cell, 'p-3 flex flex-col gap-1')}>
                                <div className="flex items-center justify-between gap-2">
                                    <span className="font-mono text-xs font-semibold truncate">{s.table}</span>
                                    <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-[var(--ifm-color-emphasis-200)] text-[var(--ifm-color-emphasis-700)] shrink-0">
                                        {s.kind}
                                    </span>
                                </div>
                                <div className="font-mono text-[11px] text-[var(--ifm-color-emphasis-700)]">{s.predicate}</div>
                                <div className="text-[10px] uppercase tracking-wide text-indigo-600 dark:text-indigo-400 mt-1">
                                    → filtered intervals
                                </div>
                            </div>
                        ))}
                        <div className="rounded-lg border border-dashed border-[var(--ifm-color-emphasis-300)] p-3 flex items-center justify-center text-center text-xs text-[var(--ifm-color-emphasis-500)]">
                            ＋ any number<br/>of tables
                        </div>
                    </div>
                </div>
            </div>

            <Down label="filtered intervals"/>

            {/* Stage 3 — REDUCE: TSAL combines the per-source filtered intervals */}
            <div className="grid grid-cols-[auto_1fr] gap-2">
                <StageLabel text="reduce"/>
                <div className={clsxJoin(cell, 'px-4 py-3 flex items-center gap-3 border-emerald-300 dark:border-emerald-500/50 bg-emerald-50/50 dark:bg-emerald-500/5')}>
                    <GitMerge className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0"/>
                    <div>
                        <div className="font-semibold text-sm">TSAL combines filtered intervals</div>
                        <div className="font-mono text-xs text-[var(--ifm-color-emphasis-700)]">
                            &amp;&nbsp;&nbsp;|&nbsp;&nbsp;~&nbsp;&nbsp;·&nbsp;&nbsp;.entity_condition()
                        </div>
                    </div>
                </div>
            </div>

            <Down/>

            {/* Output */}
            <div className={clsxJoin(cell, 'px-4 py-3 flex items-center gap-3 max-w-md mx-auto border-[var(--ifm-color-primary)]')}>
                <Target className="w-5 h-5 text-[var(--ifm-color-primary)] shrink-0"/>
                <div className="text-sm">
                    <strong>matching intervals + which entity</strong>
                    <span className="text-[var(--ifm-color-emphasis-600)]"> → event fact table / aggregations</span>
                </div>
            </div>
        </div>
    );
}

/** tiny local join helper to avoid an extra import in this self-contained demo */
function clsxJoin(...parts: string[]) {
    return parts.filter(Boolean).join(' ');
}
