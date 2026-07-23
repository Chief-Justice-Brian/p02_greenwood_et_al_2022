const countries = [
  "Argentina",
  "Australia",
  "Austria",
  "Belgium",
  "Brazil",
  "Canada",
  "Chile",
  "China",
  "Colombia",
  "Czech Republic",
  "Denmark",
  "Finland",
  "France",
  "Germany",
  "Greece",
  "Hong Kong",
  "Hungary",
  "India",
  "Indonesia",
  "Ireland",
  "Israel",
  "Italy",
  "Japan",
  "Malaysia",
  "Mexico",
  "Netherlands",
  "New Zealand",
  "Norway",
  "Peru",
  "Philippines",
  "Poland",
  "Portugal",
  "Singapore",
  "South Africa",
  "South Korea",
  "Spain",
  "Sweden",
  "Switzerland",
  "Thailand",
  "Turkey",
  "United Kingdom",
  "United States",
];

const form = document.querySelector("#simulator-form");
const countryInput = document.querySelector("#country-search");
const countryOptions = document.querySelector("#country-options");
const countryError = document.querySelector("#country-error");
const yearSelect = document.querySelector("#year");
const debtThresholdNode = document.querySelector("#debt-threshold");
const priceThresholdNode = document.querySelector("#price-threshold");
const selectedRiskNode = document.querySelector("#selected-risk");
const pointDebtNode = document.querySelector("#point-debt");
const pointPriceNode = document.querySelector("#point-price");
const zoneStatusNode = document.querySelector("#zone-status");
const summaryNode = document.querySelector("#scenario-summary");
const chartNode = document.querySelector("#risk-chart");

let currentScenario = null;
let selectedPoint = { debt: 12, price: 18 };
let activeOption = -1;

function populateYears() {
  const latestYear = new Date().getFullYear();
  for (let year = latestYear; year >= 1950; year -= 1) {
    const option = document.createElement("option");
    option.value = year;
    option.textContent = year;
    option.selected = year === 2016;
    yearSelect.append(option);
  }
}

function matchingCountries(query) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return countries;
  return countries.filter((country) =>
    country.toLowerCase().includes(normalized),
  );
}

function renderCountryOptions(query = "") {
  const matches = matchingCountries(query);
  countryOptions.replaceChildren();
  activeOption = -1;

  matches.forEach((country, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "country-option";
    button.textContent = country;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", "false");
    button.dataset.index = index;
    button.addEventListener("mousedown", (event) => {
      event.preventDefault();
      selectCountry(country);
    });
    countryOptions.append(button);
  });

  countryOptions.hidden = matches.length === 0;
  countryInput.setAttribute("aria-expanded", String(matches.length > 0));
}

function selectCountry(country) {
  countryInput.value = country;
  countryOptions.hidden = true;
  countryInput.setAttribute("aria-expanded", "false");
  countryError.textContent = "";
}

function moveActiveOption(direction) {
  const options = [...countryOptions.querySelectorAll(".country-option")];
  if (!options.length) return;
  activeOption = (activeOption + direction + options.length) % options.length;
  options.forEach((option, index) => {
    option.setAttribute("aria-selected", String(index === activeOption));
  });
  options[activeOption].scrollIntoView({ block: "nearest" });
}

countryInput.addEventListener("focus", () => {
  renderCountryOptions(countryInput.value);
});

countryInput.addEventListener("input", () => {
  countryError.textContent = "";
  renderCountryOptions(countryInput.value);
});

countryInput.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    if (countryOptions.hidden) renderCountryOptions(countryInput.value);
    moveActiveOption(1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    moveActiveOption(-1);
  } else if (event.key === "Enter" && activeOption >= 0) {
    event.preventDefault();
    const option = countryOptions.querySelectorAll(".country-option")[activeOption];
    selectCountry(option.textContent);
  } else if (event.key === "Escape") {
    countryOptions.hidden = true;
    countryInput.setAttribute("aria-expanded", "false");
  }
});

countryInput.addEventListener("blur", () => {
  window.setTimeout(() => {
    countryOptions.hidden = true;
    countryInput.setAttribute("aria-expanded", "false");
  }, 120);
});

function hashCountry(country) {
  return [...country].reduce(
    (hash, character) => (hash * 31 + character.charCodeAt(0)) % 997,
    17,
  );
}

function getScenario() {
  const country = countryInput.value.trim();
  if (!countries.includes(country)) {
    countryError.textContent = "Choose a country from the available list.";
    countryInput.focus();
    renderCountryOptions(country);
    return null;
  }

  const sector = document.querySelector('input[name="sector"]:checked').value;
  const horizon = Number(
    document.querySelector('input[name="horizon"]:checked').value,
  );
  const year = Number(yearSelect.value);
  const seed = hashCountry(country);

  // Illustrative thresholds mimic variation a real country-year dataset would
  // produce. Replace these formulas with empirical quantiles once data exists.
  const debtThreshold =
    8 + (seed % 7) * 0.55 + (sector === "Household" ? 1.4 : 0) + (year % 5) * 0.16;
  const priceThreshold =
    10 + (seed % 9) * 0.7 + (sector === "Household" ? 2.1 : 0) + (year % 4) * 0.2;

  return { country, sector, horizon, year, debtThreshold, priceThreshold };
}

function logistic(value) {
  return 1 / (1 + Math.exp(-value));
}

function riskAt(debt, price, scenario) {
  const debtSignal = (debt - scenario.debtThreshold) / 7.5;
  const priceSignal = (price - scenario.priceThreshold) / 10;
  const interaction =
    debt >= scenario.debtThreshold && price >= scenario.priceThreshold ? 0.9 : 0;
  const horizonEffect = (scenario.horizon - 1) * 0.16;
  const countryEffect = ((hashCountry(scenario.country) % 11) - 5) / 34;
  return Math.min(
    0.86,
    Math.max(
      0.015,
      logistic(
        -3.35 +
          0.72 * debtSignal +
          0.6 * priceSignal +
          interaction +
          horizonEffect +
          countryEffect,
      ),
    ),
  );
}

function axisValues(start, end, step) {
  const values = [];
  for (let value = start; value <= end; value += step) values.push(value);
  return values;
}

function renderChart(scenario, animate = true) {
  const debtValues = axisValues(-15, 35, 1);
  const priceValues = axisValues(-30, 55, 1);
  const probabilities = priceValues.map((price) =>
    debtValues.map((debt) => riskAt(debt, price, scenario) * 100),
  );

  const inRZone =
    selectedPoint.debt >= scenario.debtThreshold &&
    selectedPoint.price >= scenario.priceThreshold;
  const selectedRisk = riskAt(
    selectedPoint.debt,
    selectedPoint.price,
    scenario,
  );

  const heatmap = {
    type: "heatmap",
    x: debtValues,
    y: priceValues,
    z: probabilities,
    zmin: 0,
    zmax: 70,
    colorscale: [
      [0, "#edf5f0"],
      [0.18, "#cbe0d3"],
      [0.4, "#e8d99d"],
      [0.62, "#e9a77f"],
      [1, "#a82129"],
    ],
    colorbar: {
      title: { text: "Risk", side: "top", font: { size: 10, color: "#65736b" } },
      ticksuffix: "%",
      thickness: 9,
      len: 0.68,
      outlinewidth: 0,
      tickfont: { size: 9, color: "#65736b" },
      x: 1.015,
    },
    hovertemplate:
      "<b>%{z:.1f}% estimated risk</b><br>" +
      "Debt growth: %{x:.0f}%<br>" +
      "Price growth: %{y:.0f}%<extra></extra>",
  };

  const point = {
    type: "scatter",
    mode: "markers",
    x: [selectedPoint.debt],
    y: [selectedPoint.price],
    hoverinfo: "skip",
    marker: {
      size: 13,
      color: "#fffdf8",
      line: { color: inRZone ? "#941f24" : "#174f3e", width: 3 },
    },
    showlegend: false,
  };

  const layout = {
    margin: { t: 28, r: 65, b: 58, l: 68 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#f7f7f3",
    font: { family: "DM Sans, sans-serif", color: "#526159" },
    xaxis: {
      title: { text: "3-year debt growth (%)", standoff: 14, font: { size: 11 } },
      range: [-15, 35],
      dtick: 10,
      tickfont: { size: 10 },
      gridcolor: "rgba(255,255,255,0.35)",
      zerolinecolor: "rgba(24,37,31,0.2)",
      fixedrange: true,
    },
    yaxis: {
      title: { text: "3-year price growth (%)", standoff: 12, font: { size: 11 } },
      range: [-30, 55],
      dtick: 15,
      tickfont: { size: 10 },
      gridcolor: "rgba(255,255,255,0.35)",
      zerolinecolor: "rgba(24,37,31,0.2)",
      fixedrange: true,
    },
    shapes: [
      {
        type: "rect",
        x0: scenario.debtThreshold,
        x1: 35,
        y0: scenario.priceThreshold,
        y1: 55,
        fillcolor: "rgba(206,61,55,0.12)",
        line: { color: "#ce3d37", width: 2.5 },
        layer: "above",
      },
      {
        type: "line",
        x0: scenario.debtThreshold,
        x1: scenario.debtThreshold,
        y0: -30,
        y1: scenario.priceThreshold,
        line: { color: "#b93331", width: 1.2, dash: "dot" },
      },
      {
        type: "line",
        x0: -15,
        x1: scenario.debtThreshold,
        y0: scenario.priceThreshold,
        y1: scenario.priceThreshold,
        line: { color: "#b93331", width: 1.2, dash: "dot" },
      },
    ],
    annotations: [
      {
        x: scenario.debtThreshold + (35 - scenario.debtThreshold) / 2,
        y: scenario.priceThreshold + (55 - scenario.priceThreshold) / 2,
        text: "<b>R-ZONE</b><br><span style='font-size:9px'>JOINT TAIL</span>",
        showarrow: false,
        font: { color: "#8f2225", size: 13 },
        bgcolor: "rgba(255,253,248,0.72)",
        borderpad: 5,
      },
      {
        x: scenario.debtThreshold,
        y: -27,
        text: `<b>${scenario.debtThreshold.toFixed(1)}%</b>`,
        showarrow: false,
        font: { color: "#a62a28", size: 10 },
        bgcolor: "#fffdf8",
        bordercolor: "#ce3d37",
        borderpad: 3,
      },
      {
        x: -13.5,
        y: scenario.priceThreshold,
        text: `<b>${scenario.priceThreshold.toFixed(1)}%</b>`,
        showarrow: false,
        xanchor: "left",
        font: { color: "#a62a28", size: 10 },
        bgcolor: "#fffdf8",
        bordercolor: "#ce3d37",
        borderpad: 3,
      },
    ],
    showlegend: false,
    hoverlabel: {
      bgcolor: "#18251f",
      bordercolor: "#18251f",
      font: { color: "#fff", size: 11 },
    },
    transition: animate ? { duration: 300, easing: "cubic-in-out" } : undefined,
  };

  Plotly.react(chartNode, [heatmap, point], layout, {
    displayModeBar: false,
    responsive: true,
    scrollZoom: false,
  });

  debtThresholdNode.textContent = `≥ ${scenario.debtThreshold.toFixed(1)}%`;
  priceThresholdNode.textContent = `≥ ${scenario.priceThreshold.toFixed(1)}%`;
  selectedRiskNode.textContent = `${(selectedRisk * 100).toFixed(1)}%`;
  pointDebtNode.textContent = `${selectedPoint.debt.toFixed(1)}%`;
  pointPriceNode.textContent = `${selectedPoint.price.toFixed(1)}%`;
  summaryNode.textContent = `${scenario.country} · ${scenario.year} · ${scenario.sector} sector · ${scenario.horizon}-year forecast`;
  zoneStatusNode.textContent = inRZone ? "Inside R-Zone" : "Outside R-Zone";
  zoneStatusNode.classList.toggle("in-zone", inRZone);
}

function generate(event) {
  if (event) event.preventDefault();
  const scenario = getScenario();
  if (!scenario) return;
  currentScenario = scenario;
  selectedPoint = {
    debt: Math.round((scenario.debtThreshold - 2) * 10) / 10,
    price: Math.round((scenario.priceThreshold + 3) * 10) / 10,
  };
  renderChart(scenario);
}

form.addEventListener("submit", generate);

populateYears();
currentScenario = getScenario();
renderChart(currentScenario, false);

chartNode.on("plotly_click", (event) => {
  if (!currentScenario || !event.points?.length) return;
  selectedPoint = {
    debt: Number(event.points[0].x),
    price: Number(event.points[0].y),
  };
  renderChart(currentScenario, false);
});
