import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { listExperiments, loadExperiment, resolveImagePath } from "../../src/review_app/loader";

const tempRoots: string[] = [];

afterEach(() => {
  while (tempRoots.length) {
    fs.rmSync(tempRoots.pop(), { recursive: true, force: true });
  }
});

describe("review loader", () => {
  test("lists experiments and normalizes samples", () => {
    const root = mkFixtureRepo();
    const options = {
      repoRoot: root,
      dataRoot: path.join(root, "data"),
      experimentsRoot: path.join(root, "data", "ocr_spatial_qa", "final", "dev200"),
    };
    const experiments = listExperiments(options);
    expect(experiments).toHaveLength(1);
    expect(experiments[0]?.name).toBe("fixture_exp");

    const loaded = loadExperiment("fixture_exp", options);
    expect(loaded.samples).toHaveLength(1);
    const sample = loaded.samples[0];
    expect(sample.sampleId).toBe("img-1::img-1_0");
    expect(sample.questions).toHaveLength(1);
    expect(sample.questions[0]?.question).toContain("final dataset");
    expect(sample.tuple.answer).toBe("HELLO");
    expect(sample.tuple.qa_answer).toBe("HELLO");
    expect(sample.geometry.textCentroid?.x).toBeCloseTo(15);
    expect(sample.geometry.anchorBoxSource).toBe("derived_region");
    expect(sample.imageUrl).toContain("/api/image?path=");
  });

  test("indexes raw rows by sample_id when one ann yields multiple question types", () => {
    const root = mkFixtureRepo();
    const expDir = path.join(root, "data", "ocr_spatial_qa", "final", "dev200", "fixture_exp");
    fs.writeFileSync(
      path.join(expDir, "raw_results.jsonl"),
      [
        JSON.stringify({
          sample_id: "img-1__img-1_0__DIRECT_READ",
          ok: true,
          tuple: {
            image_id: "img-1",
            ann_id: "img-1_0",
            image_path: "data/images/example.jpg",
            image_width: 90,
            image_height: 60,
            answer: "HELLO",
            question_type: "DIRECT_READ",
            anchor_label: "upper-right area of the image",
            text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
            text_bbox: [10, 10, 10, 10],
            region_key: "ur",
          },
          items: [{ question: "What text is here?", answer: "HELLO" }],
          validations: [{ accepted: true, answer_ok: true }],
          summary: { accepted_count: 1 },
        }),
        JSON.stringify({
          sample_id: "img-1__img-1_0__YES_NO__NEG",
          ok: true,
          tuple: {
            image_id: "img-1",
            ann_id: "img-1_0",
            image_path: "data/images/example.jpg",
            image_width: 90,
            image_height: 60,
            answer: "NO",
            question_type: "YES_NO",
            anchor_label: "upper-right area of the image",
            text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
            text_bbox: [10, 10, 10, 10],
            region_key: "ur",
          },
          items: [{ question: "Does it say WORLD?", answer: "No" }],
          validations: [{ accepted: true, answer_ok: true }],
          summary: { accepted_count: 1 },
        }),
      ].join("\n") + "\n",
    );
    fs.writeFileSync(
      path.join(expDir, "ocr_qa_dataset.jsonl"),
      [
        JSON.stringify({
          sample_id: "img-1__img-1_0__DIRECT_READ",
          image_id: "img-1",
          ann_id: "img-1_0",
          image_path: "data/images/example.jpg",
          answer: "HELLO",
          question: "What text is here?",
          anchor_label: "upper-right area of the image",
          relation: "in",
          text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
          text_bbox: [10, 10, 10, 10],
          tags: { question_type: "DIRECT_READ", difficulty: "easy" },
          kd_metadata: { neighboring_text: [], nearby_anchors: [], competing_tuples: 0, image_size: [90, 60] },
          resolvable: true,
        }),
        JSON.stringify({
          sample_id: "img-1__img-1_0__YES_NO__NEG",
          image_id: "img-1",
          ann_id: "img-1_0",
          image_path: "data/images/example.jpg",
          answer: "No",
          question: "Does it say WORLD?",
          anchor_label: "upper-right area of the image",
          relation: "in",
          text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
          text_bbox: [10, 10, 10, 10],
          tags: { question_type: "YES_NO", difficulty: "medium" },
          kd_metadata: { neighboring_text: [], nearby_anchors: [], competing_tuples: 0, image_size: [90, 60] },
          resolvable: true,
        }),
      ].join("\n") + "\n",
    );

    const options = {
      repoRoot: root,
      dataRoot: path.join(root, "data"),
      experimentsRoot: path.join(root, "data", "ocr_spatial_qa", "final", "dev200"),
    };
    const loaded = loadExperiment("fixture_exp", options);
    expect(loaded.samples).toHaveLength(2);
    expect(loaded.samples[0]?.sampleId).toBe("img-1__img-1_0__DIRECT_READ");
    expect(loaded.samples[1]?.sampleId).toBe("img-1__img-1_0__YES_NO__NEG");
    expect(loaded.samples[1]?.questions[0]?.question).toContain("WORLD");
  });

  test("includes rejected raw rows for review without replacing accepted rows", () => {
    const root = mkFixtureRepo();
    const expDir = path.join(root, "data", "ocr_spatial_qa", "final", "dev200", "fixture_exp");
    fs.writeFileSync(
      path.join(expDir, "raw_results.jsonl"),
      [
        JSON.stringify({
          sample_id: "img-1__img-1_0__DIRECT_READ",
          ok: true,
          tuple: {
            image_id: "img-1",
            ann_id: "img-1_0",
            image_path: "data/images/example.jpg",
            image_width: 90,
            image_height: 60,
            answer: "HELLO",
            question_type: "DIRECT_READ",
            anchor_label: "upper-right area of the image",
            text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
            text_bbox: [10, 10, 10, 10],
            region_key: "ur",
          },
          items: [{ question: "What text is here?", answer: "HELLO" }],
          validations: [{ accepted: true, answer_ok: true }],
          summary: { accepted_count: 1 },
        }),
        JSON.stringify({
          sample_id: "img-1__img-1_1__DIRECT_READ",
          ok: true,
          tuple: {
            image_id: "img-1",
            ann_id: "img-1_1",
            image_path: "data/images/example.jpg",
            image_width: 90,
            image_height: 60,
            answer: "WORLD",
            question_type: "DIRECT_READ",
            anchor_label: "upper-right area of the image",
            text_polygon: [30, 10, 40, 10, 40, 20, 30, 20],
            text_bbox: [30, 10, 10, 10],
            region_key: "ur",
          },
          items: [{ question: "What text is on the sign?", answer: "WORLD" }],
          validations: [{ accepted: false, answer_ok: true, anchor_ok: true, length_ok: true }],
          summary: { accepted_count: 0 },
          failure_reason: "direct_read_location_missing",
          filter_stage: { reason: "direct_read_location_missing" },
        }),
      ].join("\n") + "\n",
    );
    fs.writeFileSync(
      path.join(expDir, "ocr_qa_dataset.jsonl"),
      JSON.stringify({
        sample_id: "img-1__img-1_0__DIRECT_READ",
        image_id: "img-1",
        ann_id: "img-1_0",
        image_path: "data/images/example.jpg",
        answer: "HELLO",
        question: "What text is here?",
        anchor_label: "upper-right area of the image",
        relation: "in",
        text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
        text_bbox: [10, 10, 10, 10],
        tags: { question_type: "DIRECT_READ", difficulty: "easy" },
        kd_metadata: { neighboring_text: [], nearby_anchors: [], competing_tuples: 0, image_size: [90, 60] },
        resolvable: true,
      }) + "\n",
    );

    const options = {
      repoRoot: root,
      dataRoot: path.join(root, "data"),
      experimentsRoot: path.join(root, "data", "ocr_spatial_qa", "final", "dev200"),
    };
    const loaded = loadExperiment("fixture_exp", options);
    expect(loaded.samples).toHaveLength(2);
    expect(loaded.samples[0]?.sampleStatus).toBe("accepted");
    expect(loaded.samples[1]?.sampleStatus).toBe("rejected");
    expect(loaded.samples[1]?.failureReason).toBe("direct_read_location_missing");
  });

  test("rejects image paths outside the data root", () => {
    const root = mkFixtureRepo();
    expect(() =>
      resolveImagePath("../escape.jpg", {
        repoRoot: root,
        dataRoot: path.join(root, "data"),
      }),
    ).toThrow("escapes data root");
  });
});

function mkFixtureRepo() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "sgocr-review-"));
  tempRoots.push(root);
  fs.mkdirSync(path.join(root, "data", "images"), { recursive: true });
  fs.mkdirSync(path.join(root, "data", "ocr_spatial_qa", "final", "dev200", "fixture_exp"), { recursive: true });
  fs.writeFileSync(path.join(root, "data", "images", "example.jpg"), "fixture");
  fs.writeFileSync(
    path.join(root, "data", "ocr_spatial_qa", "final", "dev200", "fixture_exp", "summary.json"),
    JSON.stringify({
      experiment: {
        provider: "gemini",
        model: "gemini-2.5-flash",
        prompt_variant: "natural_2q",
      },
      tuple_count: 1,
      qa_accept_rate: 1.0,
      tuple_full_accept_rate: 1.0,
    }),
  );
  fs.writeFileSync(
    path.join(root, "data", "ocr_spatial_qa", "final", "dev200", "fixture_exp", "raw_results.jsonl"),
    JSON.stringify({
      ok: true,
      tuple: {
        image_id: "img-1",
        ann_id: "img-1_0",
        image_path: "data/images/example.jpg",
        image_width: 90,
        image_height: 60,
        answer: "HELLO",
        text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
        text_bbox: [10, 10, 10, 10],
        region_key: "ur",
        anchor_label: "upper-right area of the image",
        relation: "in",
        unique: true,
      },
      items: [
        { question: "What text is in the top right area?", answer: "HELLO" },
        { question: "Can you read the text in the top right area?", answer: "HELLO" },
      ],
      validations: [{ accepted: true, answer_ok: true }, { accepted: true, answer_ok: true }],
      result: { usage: { totalTokenCount: 100 } },
      summary: { accepted_count: 2 },
    }) + "\n",
  );
  fs.writeFileSync(
    path.join(root, "data", "ocr_spatial_qa", "final", "dev200", "fixture_exp", "ocr_qa_dataset.jsonl"),
    JSON.stringify({
      image_id: "img-1",
      ann_id: "img-1_0",
      image_path: "data/images/example.jpg",
      answer: "HELLO",
      question: "What text is in the final dataset row?",
      anchor_label: "upper-right area of the image",
      relation: "in",
      text_polygon: [10, 10, 20, 10, 20, 20, 10, 20],
      text_bbox: [10, 10, 10, 10],
      kd_metadata: { neighboring_text: [], nearby_anchors: [], competing_tuples: 0, image_size: [90, 60] },
      resolvable: true,
    }) + "\n",
  );
  return root;
}
