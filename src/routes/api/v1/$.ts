import { createFileRoute } from "@tanstack/react-router";
import { z } from "zod";
import {
  mockBlend,
  mockExtreme,
  mockLeadTimes,
  mockRegions,
  mockReplayDetails,
  mockReplaySummaries,
  mockSources,
  mockVariables,
  mockVerification,
  mockWeightMap,
} from "@/features/data/mock";

// Same-origin gateway for the SkillBlend API.
// - If SKILLBLEND_API_URL is set (e.g. https://my-backend.example.com), requests are proxied to it.
// - Otherwise (or if the backend is unreachable / unseeded), deterministic fixtures are served
//   with an `x-skillblend-data: fixture` header so the UI can label them as demonstration data.

const region = z.enum(["KWG", "BOB", "IGP"]);
const variable = z.enum(["precipitation", "temperature", "wind_speed"]);
const season = z.enum(["winter", "pre_monsoon", "sw_monsoon", "post_monsoon"]);
const regime = z.enum(["active_monsoon", "break_monsoon", "western_disturbance", "depression", "normal"]);
const lead = z.coerce.number().int().min(1).max(720);

function json(body: unknown, status = 200, fixture = true) {
  return Response.json(body, {
    status,
    headers: { "cache-control": "no-store", ...(fixture ? { "x-skillblend-data": "fixture" } : {}) },
  });
}

function fixtureFor(path: string, q: URLSearchParams): Response {
  const p = (k: string) => q.get(k) ?? undefined;
  try {
    switch (path) {
      case "health":
        return json({ status: "ok", mode: "fixture" });
      case "regions":
        return json(mockRegions);
      case "variables":
        return json(mockVariables);
      case "lead-times":
        return json(mockLeadTimes);
      case "sources":
        return json(mockSources);
      case "forecast/blend":
        return json(mockBlend(region.parse(p("region")), variable.parse(p("variable")), lead.parse(p("lead_hours"))));
      case "weights/map":
        return json(
          mockWeightMap(
            region.parse(p("region")),
            variable.parse(p("variable")),
            season.parse(p("season")),
            regime.parse(p("regime")),
          ),
        );
      case "skill/verification":
        return json(mockVerification);
      case "extreme/guidance":
        return json(mockExtreme(region.parse(p("region")), lead.parse(p("lead_hours"))));
      case "replay/events":
        return json(mockReplaySummaries);
    }
    if (path.startsWith("replay/events/")) {
      const id = decodeURIComponent(path.slice("replay/events/".length));
      const match = mockReplayDetails.find((e) => e.event_id === id);
      return match ? json(match) : json({ detail: "Replay event not found" }, 404);
    }
    return json({ detail: "Unknown endpoint" }, 404);
  } catch {
    return json({ detail: "Invalid query parameters" }, 422);
  }
}

async function handle({ request, params }: { request: Request; params: { _splat?: string } }) {
  const path = (params._splat ?? "").replace(/^\/+|\/+$/g, "");
  const url = new URL(request.url);
  const backend = process.env["SKILLBLEND_API_URL"];

  if (backend) {
    try {
      const upstream = await fetch(`${backend.replace(/\/+$/, "")}/api/v1/${path}${url.search}`, {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(8000),
      });
      if (upstream.ok || upstream.status === 404 || upstream.status === 422) {
        return new Response(upstream.body, {
          status: upstream.status,
          headers: { "content-type": upstream.headers.get("content-type") ?? "application/json", "cache-control": "no-store" },
        });
      }
      // 5xx / 503 (e.g. unseeded backend) → fall through to fixtures
    } catch {
      // unreachable backend → fall through to fixtures
    }
  }
  return fixtureFor(path, url.searchParams);
}

export const Route = createFileRoute("/api/v1/$")({
  server: {
    handlers: {
      GET: handle,
    },
  },
});
