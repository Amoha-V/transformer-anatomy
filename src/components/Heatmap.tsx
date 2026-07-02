import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { Matrix } from '../model/types';

interface HeatmapProps {
  matrix: Matrix;
  rowLabels?: string[];
  colLabels?: string[];           // if provided, rendered rotated along the top (for square attention matrices)
  mode: 'diverging' | 'sequential'; // diverging: teal<0<amber (raw values); sequential: paper->amber (0..1, e.g. attention weights)
  causalMask?: boolean;            // shade cells where col > row as "masked / never attended to"
  highlightRow?: number | null;
  onHoverRow?: (row: number | null) => void;
  cellWidth?: number;
  cellHeight?: number;
}

const PAPER = [239, 241, 235];
const AMBER = [224, 138, 46];
const TEAL = [42, 127, 126];

function rgbStr([r, g, b]: number[]): string {
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

export function Heatmap({
  matrix, rowLabels, colLabels, mode, causalMask, highlightRow = null, onHoverRow,
  cellWidth = 16, cellHeight = 17,
}: HeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || matrix.length === 0) return;
    const rows = matrix.length, cols = matrix[0].length;
    const gutter = rowLabels ? 64 : 0;
    const topGutter = colLabels ? 56 : 0;
    const W = gutter + cols * cellWidth;
    const H = topGutter + rows * cellHeight;

    const svg = d3.select(svgRef.current)
      .attr('width', W).attr('height', H)
      .attr('viewBox', `0 0 ${W} ${H}`);

    let colorScale: (v: number) => string;
    if (mode === 'diverging') {
      let maxAbs = 0;
      for (const r of matrix) for (const v of r) maxAbs = Math.max(maxAbs, Math.abs(v));
      maxAbs = maxAbs || 1;
      colorScale = (v: number) => {
        const t = Math.max(-1, Math.min(1, v / maxAbs));
        const target = t >= 0 ? AMBER : TEAL;
        const tt = Math.abs(t);
        return rgbStr(PAPER.map((p, i) => p + (target[i] - p) * tt));
      };
    } else {
      colorScale = (v: number) => {
        const t = Math.max(0, Math.min(1, v));
        return rgbStr(PAPER.map((p, i) => p + (AMBER[i] - p) * t));
      };
    }

    // flatten to [row, col, value]
    const cells: { row: number; col: number; v: number }[] = [];
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) cells.push({ row: i, col: j, v: matrix[i][j] });

    const g = svg.selectAll<SVGGElement, null>('g.cells').data([null]).join('g').attr('class', 'cells');

    g.selectAll<SVGRectElement, typeof cells[0]>('rect')
      .data(cells, (d) => `${d.row}-${d.col}`)
      .join('rect')
      .attr('x', (d) => gutter + d.col * cellWidth)
      .attr('y', (d) => topGutter + d.row * cellHeight)
      .attr('width', cellWidth - 1)
      .attr('height', cellHeight - 1)
      .attr('fill', (d) => {
        if (causalMask && d.col > d.row) return '#dcd9d3';
        return colorScale(d.v);
      })
      .attr('stroke', (d) => (causalMask && d.col > d.row ? '#c9b8b0' : 'none'))
      .attr('stroke-width', 1)
      .attr('data-row', (d) => d.row);

    // causal-mask hatch lines
    if (causalMask) {
      const hatchCells = cells.filter((d) => d.col > d.row);
      svg.selectAll('line.hatch').data(hatchCells, (d: any) => `${d.row}-${d.col}`).join('line')
        .attr('class', 'hatch')
        .attr('x1', (d) => gutter + d.col * cellWidth)
        .attr('y1', (d) => topGutter + d.row * cellHeight)
        .attr('x2', (d) => gutter + d.col * cellWidth + cellWidth - 1)
        .attr('y2', (d) => topGutter + d.row * cellHeight + cellHeight - 1)
        .attr('stroke', '#c9b8b0').attr('stroke-width', 1);
    }

    // row labels
    if (rowLabels) {
      svg.selectAll('text.rowlabel').data(rowLabels).join('text')
        .attr('class', 'rowlabel')
        .attr('x', gutter - 8)
        .attr('y', (_, i) => topGutter + i * cellHeight + cellHeight / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('font-family', "'JetBrains Mono', monospace")
        .attr('font-size', 11)
        .attr('font-weight', (_, i) => (i === highlightRow ? 600 : 400))
        .attr('fill', (_, i) => (i === highlightRow ? '#b86a18' : '#1b2430'))
        .text((d) => (d.length > 9 ? d.slice(0, 9) : d));
    }

    // col labels (rotated)
    if (colLabels) {
      const sel = svg.selectAll('g.collabel').data(colLabels).join('g')
        .attr('class', 'collabel')
        .attr('transform', (_, j) => `translate(${gutter + j * cellWidth + cellWidth / 2}, ${topGutter - 8}) rotate(-60)`);
      sel.selectAll('text').data((d) => [d]).join('text')
        .attr('text-anchor', 'start')
        .attr('dominant-baseline', 'middle')
        .attr('font-family', "'JetBrains Mono', monospace")
        .attr('font-size', 10.5)
        .attr('fill', '#1b2430')
        .text((d) => (d.length > 9 ? d.slice(0, 9) : d));
    }

    // highlight outline for the hovered/selected row
    svg.selectAll('rect.highlight-outline').data(highlightRow != null ? [highlightRow] : []).join('rect')
      .attr('class', 'highlight-outline')
      .attr('x', gutter + 0.5)
      .attr('y', (d) => topGutter + d * cellHeight + 0.5)
      .attr('width', cols * cellWidth - 1)
      .attr('height', cellHeight - 1)
      .attr('fill', 'none')
      .attr('stroke', '#b86a18')
      .attr('stroke-width', 2);

    // hover handling
    if (onHoverRow) {
      svg.on('mousemove', (event) => {
        const [, y] = d3.pointer(event);
        if (y < topGutter) return;
        const row = Math.floor((y - topGutter) / cellHeight);
        if (row >= 0 && row < rows) onHoverRow(row);
      });
      svg.on('mouseleave', () => onHoverRow(null));
    }
  }, [matrix, rowLabels, colLabels, mode, causalMask, highlightRow, cellWidth, cellHeight, onHoverRow]);

  return <svg ref={svgRef} style={{ display: 'block' }} />;
}
