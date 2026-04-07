export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export type Point = {
  x: number;
  y: number;
};

export type DisplayBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
  cx: number;
  cy: number;
};

export type ReviewValidation = {
  answer_ok?: boolean;
  anchor_ok?: boolean;
  length_ok?: boolean;
  duplicate_ok?: boolean;
  accepted?: boolean;
};

export type ReviewQuestion = {
  index: number;
  question: string;
  answer: string;
  validation: ReviewValidation | null;
};

export type ReviewTupleGeometry = {
  textPolygon: Point[];
  textCentroid: Point | null;
  textBox: DisplayBox | null;
  anchorBox: DisplayBox | null;
  anchorCentroid: Point | null;
  anchorBoxSource: string | null;
  queryAnchorBox: DisplayBox | null;
  queryAnchorCentroid: Point | null;
  refBox: DisplayBox | null;
  refCentroid: Point | null;
};

export type ReviewSample = {
  sampleId: string;
  sampleIndex: number;
  sampleStatus: "accepted" | "rejected";
  failureReason: string | null;
  imageId: string;
  annId: string | null;
  imagePath: string | null;
  imageUrl: string | null;
  imageWidth: number | null;
  imageHeight: number | null;
  tuple: Record<string, JsonValue>;
  geometry: ReviewTupleGeometry;
  questions: ReviewQuestion[];
  usage: Record<string, JsonValue> | null;
  summary: Record<string, JsonValue> | null;
  filterStage: Record<string, JsonValue> | null;
  ok: boolean;
};

export type ExperimentSummary = {
  name: string;
  path: string;
  sampleCount: number | null;
  qaAcceptRate: number | null;
  tupleAcceptRate: number | null;
  model: string | null;
  provider: string | null;
  promptVariant: string | null;
  inputTupleCount: number | null;
  resolvableTupleCount: number | null;
  droppedTupleCount: number | null;
  generatedQas: number | null;
  acceptedQas: number | null;
  sameImageUniverseCount: number | null;
  resolvabilityStats: Record<string, JsonValue> | null;
  stageCounts?: Record<string, JsonValue> | null;
  questionTypeCounts?: Record<string, JsonValue> | null;
  selectedQuestionTypeCounts?: Record<string, JsonValue> | null;
  disabledQuestionTypes?: JsonValue[] | null;
  failureCounts?: Record<string, JsonValue> | null;
};

export type AuditQuestionJudgment = "yes" | "no" | "skip" | "";
export type AuditTernary = "yes" | "no" | "marginal" | "";
export type AuditBinary = "yes" | "no" | "";

export type AuditEntry = {
  sampleId: string;
  sampleIndex: number;
  imageId: string;
  annId: string | null;
  ocrCorrect: AuditBinary;
  anchorGrounding: AuditTernary;
  spatialRelation: AuditBinary;
  questionNatural?: AuditBinary;
  answerCorrect?: AuditBinary;
  overallGo: AuditTernary;
  questionFaithful: AuditQuestionJudgment[];
  tagsSnapshot?: Record<string, JsonValue> | null;
  notes: string;
  updatedAt: string;
};
