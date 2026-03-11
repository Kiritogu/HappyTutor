"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { apiUrl } from "@/lib/api";
import { getAccessToken } from "@/lib/auth";
import { useTranslation } from "react-i18next";

type GraphNode = {
  id: string;
  label: string;
  type: string;
  score?: number;
};

type GraphEdge = {
  source: string;
  target: string;
  relation: string;
  weight?: number;
};

type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
};

export default function GraphPage() {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  const [kbName, setKbName] = useState("ai_textbook");
  const [query, setQuery] = useState("");
  const [hops, setHops] = useState(2);
  const [limit, setLimit] = useState(120);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<GraphPayload>({ nodes: [], edges: [], truncated: false });

  const elements = useMemo<ElementDefinition[]>(() => {
    const nodes = data.nodes.map((n) => ({
      data: { id: n.id, label: n.label, type: n.type, score: n.score ?? 0 },
    }));
    const edges = data.edges.map((e, idx) => ({
      data: {
        id: `${e.source}-${e.target}-${idx}`,
        source: e.source,
        target: e.target,
        relation: e.relation,
        weight: e.weight ?? 1,
      },
    }));
    return [...nodes, ...edges];
  }, [data.edges, data.nodes]);

  useEffect(() => {
    if (!canvasRef.current) return;

    if (!cyRef.current) {
      cyRef.current = cytoscape({
        container: canvasRef.current,
        style: [
          {
            selector: "node",
            style: {
              "background-color": "#0f766e",
              label: "data(label)",
              color: "#1f2937",
              "font-size": "10px",
              "text-valign": "bottom",
              "text-wrap": "wrap",
              "text-max-width": "120px",
            },
          },
          {
            selector: "edge",
            style: {
              width: 1.5,
              "line-color": "#94a3b8",
              "target-arrow-color": "#94a3b8",
              "target-arrow-shape": "triangle",
              "curve-style": "bezier",
              label: "data(relation)",
              "font-size": "8px",
              color: "#64748b",
            },
          },
          {
            selector: "node[type = 'keyword']",
            style: { "background-color": "#2563eb" },
          },
          {
            selector: "node[type = 'concept']",
            style: { "background-color": "#7c3aed" },
          },
        ],
        layout: { name: "cose", animate: true, fit: true },
      });
      return;
    }

    const cy = cyRef.current;
    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: "cose", animate: true, fit: true }).run();
  }, [elements]);

  useEffect(() => {
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, []);

  const runQuery = async () => {
    if (!query.trim()) {
      setError(t("Please input a query"));
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await fetch(
        apiUrl(
          `/api/v1/graph/subgraph?kb_name=${encodeURIComponent(kbName)}&q=${encodeURIComponent(query)}&hops=${hops}&limit=${limit}`,
        ),
        {
          headers: {
            Authorization: `Bearer ${getAccessToken() ?? ""}`,
          },
        },
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const json = (await res.json()) as GraphPayload;
      setData({
        nodes: json.nodes || [],
        edges: json.edges || [],
        truncated: Boolean(json.truncated),
      });
    } catch (e) {
      setError(`${t("Graph query failed")}: ${String(e)}`);
      setData({ nodes: [], edges: [], truncated: false });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen p-4 flex flex-col gap-4 bg-slate-50 dark:bg-slate-900">
      <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4 grid grid-cols-1 md:grid-cols-6 gap-3">
        <input
          value={kbName}
          onChange={(e) => setKbName(e.target.value)}
          className="md:col-span-1 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-transparent"
          placeholder={t("Knowledge base")}
        />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="md:col-span-3 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-transparent"
          placeholder={t("Search graph anchor...")}
        />
        <input
          type="number"
          min={1}
          max={3}
          value={hops}
          onChange={(e) => setHops(Number(e.target.value))}
          className="md:col-span-1 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-transparent"
        />
        <input
          type="number"
          min={1}
          max={300}
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="md:col-span-1 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-transparent"
        />
        <button
          onClick={runQuery}
          disabled={loading}
          className="md:col-span-6 px-4 py-2 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-60"
        >
          {loading ? t("Loading...") : t("Run Graph Query")}
        </button>
        {error && <p className="md:col-span-6 text-sm text-red-500">{error}</p>}
        {data.truncated && (
          <p className="md:col-span-6 text-sm text-amber-600">
            {t("Result truncated. Narrow query or lower limit for precise exploration.")}
          </p>
        )}
      </div>

      <div className="flex-1 min-h-0 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
        <div className="h-full" ref={canvasRef} />
      </div>
    </div>
  );
}

