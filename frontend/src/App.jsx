import { useEffect, useMemo, useState } from "react";

const COMPONENT_LABELS = {
  ic: "IC",
  resistor: "Resistor",
  capacitor: "Capacitor",
  diode: "Diode",
};

function overlayStyle(box, imageSize) {
  if (!box || !imageSize?.width || !imageSize?.height) {
    return { display: "none" };
  }
  return {
    left: `${(box.x / imageSize.width) * 100}%`,
    top: `${(box.y / imageSize.height) * 100}%`,
    width: `${(box.width / imageSize.width) * 100}%`,
    height: `${(box.height / imageSize.height) * 100}%`,
  };
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed";
  } catch {
    return `Request failed (${response.status})`;
  }
}

export default function App() {
  const [boards, setBoards] = useState([]);
  const [defects, setDefects] = useState([]);
  const [boardId, setBoardId] = useState("");
  const [component, setComponent] = useState("");
  const [instance, setInstance] = useState(1);
  const [defectId, setDefectId] = useState("");
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [boardResponse, defectResponse] = await Promise.all([
          fetch("/api/boards"),
          fetch("/api/defects"),
        ]);
        if (!boardResponse.ok) throw new Error(await readError(boardResponse));
        if (!defectResponse.ok) throw new Error(await readError(defectResponse));

        const boardData = await boardResponse.json();
        const defectData = await defectResponse.json();
        const firstBoard =
          boardData.find((board) => board.annotated) || boardData[0];

        setBoards(boardData);
        setDefects(defectData);
        if (firstBoard) {
          setBoardId(firstBoard.id);
          if (firstBoard.components[0]) {
            setComponent(firstBoard.components[0].component);
            setInstance(firstBoard.components[0].instance);
          }
        }
        if (defectData[0]) setDefectId(defectData[0].id);
      } catch (loadError) {
        setError(
          loadError.message ||
            "Could not reach the API. Start the FastAPI server on port 8000."
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const selectedBoard = useMemo(
    () => boards.find((board) => board.id === boardId),
    [boards, boardId]
  );
  const componentOptions = selectedBoard?.components || [];
  const selectedPart = componentOptions.find(
    (item) => item.component === component && item.instance === instance
  );
  const selectedDefect = defects.find((item) => item.id === defectId);

  useEffect(() => {
    if (!selectedBoard) return;
    const valid = selectedBoard.components.some(
      (item) => item.component === component && item.instance === instance
    );
    if (!valid && selectedBoard.components[0]) {
      setComponent(selectedBoard.components[0].component);
      setInstance(selectedBoard.components[0].instance);
    }
  }, [selectedBoard, component, instance]);

  async function handleGenerate() {
    setError("");
    setGenerating(true);
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          board: boardId,
          component,
          instance,
          defect: defectId,
        }),
      });
      if (!response.ok) throw new Error(await readError(response));
      setResult(await response.json());
    } catch (generateError) {
      setError(generateError.message || "Could not generate the test image.");
    } finally {
      setGenerating(false);
    }
  }

  const originalSrc = boardId
    ? `/api/boards/${boardId}/image?max_width=1400`
    : "";
  const generatedSrc = result
    ? `${result.imageUrl}?max_width=1600&t=${encodeURIComponent(result.filename)}`
    : "";

  return (
    <div className="page">
      <header className="hero">
        <h1>PCBA placement fault simulator</h1>
        <p className="lede">
          Create a realistic defective test image from a good board. Choose a
          part and a placement fault, then generate a labeled sample for AOI
          inspection demos.
        </p>
      </header>

      {error ? <div className="banner error">{error}</div> : null}
      {loading ? <div className="banner">Loading boards…</div> : null}

      <section className="panel">
        <div className="step">
          <h2>1. Board</h2>
          <div className="chip-row">
            {boards.map((board) => (
              <button
                key={board.id}
                className={board.id === boardId ? "chip active" : "chip"}
                disabled={!board.annotated}
                onClick={() => {
                  setBoardId(board.id);
                  setResult(null);
                }}
              >
                {board.id}
                {!board.annotated ? <span className="hint">no boxes</span> : null}
              </button>
            ))}
          </div>
        </div>

        <div className="step">
          <h2>2. Component</h2>
          <div className="chip-row">
            {componentOptions.map((item) => {
              const selected =
                item.component === component && item.instance === instance;
              const label = COMPONENT_LABELS[item.component] || item.component;
              const duplicates = componentOptions.filter(
                (entry) => entry.component === item.component
              ).length;
              return (
                <button
                  key={`${item.component}-${item.instance}`}
                  className={selected ? "chip active" : "chip"}
                  onClick={() => {
                    setComponent(item.component);
                    setInstance(item.instance);
                    setResult(null);
                  }}
                >
                  {label}
                  {duplicates > 1 ? ` #${item.instance}` : ""}
                </button>
              );
            })}
            {!componentOptions.length ? (
              <p className="muted">This board has no annotations yet.</p>
            ) : null}
          </div>
        </div>

        <div className="step">
          <h2>3. Placement fault</h2>
          <div className="defect-grid">
            {defects.map((defect) => (
              <button
                key={defect.id}
                className={defect.id === defectId ? "defect active" : "defect"}
                onClick={() => {
                  setDefectId(defect.id);
                  setResult(null);
                }}
              >
                <strong>{defect.label}</strong>
                <span>{defect.industry}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="actions">
          <button
            className="primary"
            disabled={
              generating || !selectedBoard?.annotated || !component || !defectId
            }
            onClick={handleGenerate}
          >
            {generating
              ? "Generating test image…"
              : "Generate defective test image"}
          </button>
          {selectedDefect ? (
            <p className="muted">
              Will apply{" "}
              <strong>{selectedDefect.industry.toLowerCase()}</strong> to the
              selected {COMPONENT_LABELS[component] || component}.
            </p>
          ) : null}
        </div>
      </section>

      <section className="compare">
        <article className="frame">
          <div className="frame-head">
            <h2>Good board</h2>
            <span>{boardId || "—"}</span>
          </div>
          <div className="stage">
            {originalSrc ? (
              <img src={originalSrc} alt={`Original ${boardId}`} />
            ) : (
              <div className="empty">Select a board</div>
            )}
            {selectedPart && selectedBoard ? (
              <div
                className="box original"
                style={overlayStyle(selectedPart, selectedBoard)}
              />
            ) : null}
          </div>
        </article>

        <article className="frame">
          <div className="frame-head">
            <h2>Generated defect</h2>
            <span>{result ? result.defect.label : "Waiting"}</span>
          </div>
          <div className="stage">
            {generatedSrc ? (
              <img src={generatedSrc} alt="Generated defective PCB" />
            ) : (
              <div className="empty">
                Generate a test image to show visitors a known placement fault.
              </div>
            )}
          </div>
          {result ? (
            <p className="caption">
              {COMPONENT_LABELS[result.component] || result.component}:{" "}
              {result.defect.industry}. File {result.filename}
            </p>
          ) : null}
        </article>
      </section>
    </div>
  );
}
