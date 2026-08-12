import { describe, expect, it } from "vitest";
import { formatTimecode } from "./timecode";

describe("formatTimecode", () => {
  it.each([
    [0, "0:00"],
    [5, "0:05"],
    [14, "0:14"],
    [59, "0:59"],
    [60, "1:00"],
    [75, "1:15"],
    [600, "10:00"],
    [3661, "61:01"],
  ])("%s seconds reads %s", (input, expected) => {
    expect(formatTimecode(input)).toBe(expected);
  });

  it("floors fractional seconds rather than rounding up past the moment", () => {
    expect(formatTimecode(14.9)).toBe("0:14");
  });

  it("degrades safely on nonsense input", () => {
    expect(formatTimecode(-1)).toBe("0:00");
    expect(formatTimecode(Number.NaN)).toBe("0:00");
    expect(formatTimecode(Number.POSITIVE_INFINITY)).toBe("0:00");
  });
});
