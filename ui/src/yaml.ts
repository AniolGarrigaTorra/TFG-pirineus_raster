function isScalar(value: unknown): boolean {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function scalar(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number" || typeof value === "boolean") return String(value);

  const text = String(value);
  if (/^[A-Za-z0-9_./-]+$/.test(text) && text !== "") {
    return text;
  }
  return JSON.stringify(text);
}

function renderArray(items: unknown[], indent: number): string {
  if (items.length === 0) return "[]";

  if (items.every(isScalar)) {
    return `[${items.map(scalar).join(", ")}]`;
  }

  return items
    .map((item) => {
      if (isScalar(item)) {
        return `${" ".repeat(indent)}- ${scalar(item)}`;
      }
      const rendered = renderYaml(item, indent + 2).trimEnd();
      return `${" ".repeat(indent)}-\n${rendered}`;
    })
    .join("\n");
}

export function renderYaml(value: unknown, indent = 0): string {
  if (Array.isArray(value)) {
    return renderArray(value, indent) + "\n";
  }

  if (isScalar(value)) {
    return `${" ".repeat(indent)}${scalar(value)}\n`;
  }

  const obj = value as Record<string, unknown>;
  const lines: string[] = [];

  for (const [key, item] of Object.entries(obj)) {
    if (item === undefined) continue;

    const pad = " ".repeat(indent);

    if (Array.isArray(item)) {
      if (item.length === 0 || item.every(isScalar)) {
        lines.push(`${pad}${key}: ${renderArray(item, indent + 2)}`);
      } else {
        lines.push(`${pad}${key}:`);
        lines.push(renderArray(item, indent + 2));
      }
      continue;
    }

    if (isScalar(item)) {
      lines.push(`${pad}${key}: ${scalar(item)}`);
      continue;
    }

    lines.push(`${pad}${key}:`);
    lines.push(renderYaml(item, indent + 2).trimEnd());
  }

  return `${lines.join("\n")}\n`;
}

