// Shared artifact-info-panel logic used by both the 3D model viewer
// (viewer.js) and the MISHA spectral-band viewer (msi.js). Both pages show
// the same info-panel markup (#infoPanelWrapper/#infoToggle/#infoPanel/
// #infoName/#infoType/#infoDescription/#infoLink/#crossLink) and both need
// to fetch the shared artifacts/manifest.json, find this page's artifact,
// and populate that panel — this used to be duplicated per page.

export async function fetchArtifact(artifactId) {
  const res = await fetch("artifacts/manifest.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`manifest request failed: ${res.status}`);
  const data = await res.json();
  return (data.artifacts || []).find((a) => a.id === artifactId);
}

export function titleCase(str) {
  return str.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
}

// Wires the info panel's open/closed toggle. Call once per page.
export function initInfoToggle() {
  const infoToggle = document.getElementById("infoToggle");
  const infoPanel = document.getElementById("infoPanel");
  if (!infoToggle || !infoPanel) return;
  infoToggle.addEventListener("click", () => {
    const open = infoPanel.classList.toggle("open");
    infoToggle.innerHTML = open ? "Artifact Info &#x25B2;" : "Artifact Info &#x25BC;";
  });
}

// Populates #infoName/#infoType/#infoDescription/#infoLink from a manifest
// artifact entry and sets the page title. titleSuffix differs per page
// ("Artifact Viewer" vs "Spectral Bands").
export function populateInfoPanel(artifact, titleSuffix) {
  document.title = `${artifact.name} — ${titleSuffix}`;

  const infoName = document.getElementById("infoName");
  const infoType = document.getElementById("infoType");
  const infoDescription = document.getElementById("infoDescription");
  const infoLink = document.getElementById("infoLink");

  if (infoName) infoName.textContent = artifact.name;
  if (infoType) infoType.textContent = titleCase(artifact.type || "other");
  if (infoDescription) infoDescription.textContent = artifact.description || "";
  if (infoLink) {
    if (artifact.link) {
      infoLink.href = artifact.link;
      infoLink.textContent = `${artifact.linkLabel || "View Source"} ↗`;
      infoLink.style.display = "inline-block";
    } else {
      infoLink.style.display = "none";
    }
  }
}

// Cross-link between the 3D viewer and the MSI band viewer, shown only when
// the artifact actually has the OTHER kind of data. `targetView` is which
// view this link should point at from the CURRENT page — "msi" from
// viewer.html, "model" from msi.html.
export function populateCrossLink(artifact, targetView) {
  const crossLink = document.getElementById("crossLink");
  if (!crossLink) return;

  if (targetView === "msi" && artifact.msi) {
    crossLink.href = `msi.html?id=${encodeURIComponent(artifact.id)}`;
    crossLink.textContent = "View Spectral Bands →";
    crossLink.style.display = "inline-block";
  } else if (targetView === "model" && artifact.model) {
    crossLink.href = `viewer.html?id=${encodeURIComponent(artifact.id)}`;
    crossLink.textContent = "View 3D Model →";
    crossLink.style.display = "inline-block";
  } else {
    crossLink.style.display = "none";
  }
}
