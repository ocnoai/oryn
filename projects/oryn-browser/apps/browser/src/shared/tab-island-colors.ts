export const TAB_ISLAND_COLORS = [
  { name: 'Lavender', value: '#8b8cff' },
  { name: 'Blue', value: '#67a9ff' },
  { name: 'Cyan', value: '#4cc9f0' },
  { name: 'Teal', value: '#53c7a2' },
  { name: 'Green', value: '#7bc96f' },
  { name: 'Lime', value: '#b1d45a' },
  { name: 'Amber', value: '#e5bd55' },
  { name: 'Orange', value: '#ffab6b' },
  { name: 'Coral', value: '#f27777' },
  { name: 'Pink', value: '#e77dab' },
] as const;

export function isTabIslandColor(value: string): boolean {
  const normalized = value.toLowerCase();
  return TAB_ISLAND_COLORS.some((color) => color.value === normalized);
}
