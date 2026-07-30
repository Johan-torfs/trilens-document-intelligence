export const DOCUMENT_TYPES = [
  { value: "", label: "Alle types" },
  { value: "invoice", label: "Invoice" },
  { value: "purchase_order", label: "Purchase order" },
  { value: "receipt", label: "Receipt" },
  { value: "delivery_note", label: "Delivery note" },
  { value: "application_form", label: "Application form" },
  { value: "identity_card", label: "Identity card" },
] as const;

export const ANALYSIS_QUESTIONS = [
  "Describe the visual structure and layout of this document.",
  "Is there a signature visible?",
  "Which visual elements suggest this is an invoice?",
  "Does this document appear to contain multiple product rows?",
] as const;
