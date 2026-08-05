from pathlib import Path

from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.analysis_service import AnalysisService
from app.services.document_classification_service import (
    DocumentClassificationService,
)
from app.services.document_intelligence_pipeline import (
    DocumentIntelligencePipeline,
)
from app.services.document_preparation_service import (
    DocumentPreparationService,
)
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
from app.services.score_calibration import LinearScoreCalibrator
from app.services.text_indexing_service import TextIndexingService
from app.strategies.image_document import ImageDocumentStrategy
from app.strategies.sentence_embedding import SentenceEmbeddingStrategy
from app.strategies.pdf_document import PDFDocumentStrategy
from app.strategies.siglip_embedding import SiglipEmbeddingStrategy
from app.strategies.open_flamingo_analysis import (
    OpenFlamingoAnalysisStrategy,
)
from app.services.document_ocr_service import DocumentOCRService
from app.strategies.doctr_ocr import DocTROCRStrategy
from app.domain.classification import DocumentTypeCandidate


DOCUMENT_TYPE_CANDIDATES: list[DocumentTypeCandidate] = [
    DocumentTypeCandidate(
        label="invoice",
        prompts=("a business invoice", "an invoice document", "a billing statement"),
        keywords=("invoice", "invoice no", "invoice number", "bill to", "total due",
                  "amount due", "subtotal", "vat", "payment terms", "due date"),
    ),
    DocumentTypeCandidate(
        label="purchase_order",
        prompts=("a purchase order", "a procurement document", "a supplier order form"),
        keywords=("purchase order", "po number", "order number", "ship to", "vendor",
                  "buyer", "ordered by", "delivery date", "order date"),
    ),
    DocumentTypeCandidate(
        label="receipt",
        prompts=("a receipt", "a payment receipt", "a store receipt"),
        keywords=("receipt", "cash receipt", "total paid", "thank you for your purchase",
                  "change", "cashier", "transaction id"),
    ),
    DocumentTypeCandidate(
        label="delivery_note",
        prompts=("a delivery note", "a packing slip", "a dispatch note"),
        keywords=("delivery note", "packing slip", "dispatch note", "consignment",
                  "shipped", "delivered to", "goods received", "items delivered"),
    ),
    DocumentTypeCandidate(
        label="application_form",
        prompts=("an application form", "a filled registration form", "a form document"),
        keywords=("application form", "applicant", "date of birth", "please complete",
                  "full name", "address", "signature", "submit"),
    ),
    DocumentTypeCandidate(
        label="identity_card",
        prompts=("an identity card", "a national id card", "a personal identification card"),
        keywords=("identity card", "id card", "national identity", "date of birth",
                  "nationality", "place of birth", "valid until", "card number"),
    ),
    DocumentTypeCandidate(
        label="contract",
        prompts=("a contract", "a legal agreement", "a signed contract document"),
        keywords=("contract", "agreement", "whereas", "hereby", "obligations",
                  "terms and conditions", "signed", "witness", "effective date"),
    ),
    DocumentTypeCandidate(
        label="letter",
        prompts=("a formal business letter", "a letter", "a correspondence document"),
        keywords=("dear", "sincerely", "yours faithfully", "kind regards",
                  "to whom it may concern", "yours truly"),
    ),
    DocumentTypeCandidate(
        label="report",
        prompts=("a report document", "a business report", "an annual report"),
        keywords=("report", "executive summary", "findings", "conclusion",
                  "recommendations", "prepared by", "overview", "analysis"),
    ),
    DocumentTypeCandidate(
        label="bank_statement",
        prompts=("a bank statement", "an account statement", "a financial account summary"),
        keywords=("bank statement", "account number", "balance", "deposit",
                  "withdrawal", "opening balance", "closing balance", "statement period"),
    ),
    DocumentTypeCandidate(
        label="pay_slip",
        prompts=("a payslip", "a salary slip", "a pay stub"),
        keywords=("pay slip", "payslip", "salary", "gross pay", "net pay",
                  "deductions", "employee", "employer", "pay period"),
    ),
    DocumentTypeCandidate(
        label="quotation",
        prompts=("a price quotation", "a quote document", "an estimate"),
        keywords=("quotation", "quote", "estimate", "unit price", "total price",
                  "discount", "quote number", "valid until", "offer"),
    ),
    DocumentTypeCandidate(
        label="certificate",
        prompts=("a certificate", "a certificate of completion", "an award certificate"),
        keywords=("certificate", "awarded to", "certify", "this is to certify",
                  "completion", "achievement", "issued by", "diploma"),
    ),
    DocumentTypeCandidate(
        label="tax_document",
        prompts=("a tax form", "a tax return document", "a fiscal document"),
        keywords=("tax", "vat return", "tax return", "fiscal year", "taxable income",
                  "tax reference", "withholding", "deductions"),
    ),
]


def create_document_intelligence_pipeline(
    document_repository: DocumentRepository,
    vector_repository: VectorRepository,
    upload_dir: Path,
    open_flamingo_enabled: bool = False,
    open_flamingo_device: str | None = "cpu",
) -> DocumentIntelligencePipeline:
    preparation_service = DocumentPreparationService(
        strategies=[
            ImageDocumentStrategy(),
            PDFDocumentStrategy(),
        ]
    )

    retrieval_strategy = SiglipEmbeddingStrategy()
    text_strategy = SentenceEmbeddingStrategy()
    ocr_strategy = DocTROCRStrategy()

    indexing_service = IndexingService(
        strategy=retrieval_strategy,
        vector_repository=vector_repository,
    )

    retrieval_service = RetrievalService(
        strategy=retrieval_strategy,
        vector_repository=vector_repository,
        text_strategy=text_strategy,
    )

    document_search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    analysis_strategy = OpenFlamingoAnalysisStrategy(
        device=open_flamingo_device,
    )

    analysis_service = AnalysisService(
        strategy=analysis_strategy,
        enabled=open_flamingo_enabled,
    )

    ocr_service = DocumentOCRService(
        strategy=ocr_strategy,
        document_repository=document_repository,
    )

    visual_score_calibrator = LinearScoreCalibrator(
        noise_floor=0.04,
        ceiling=0.28,
    )

    text_indexing_service = TextIndexingService(
        strategy=text_strategy,
        vector_repository=vector_repository,
    )

    classification_service = DocumentClassificationService(
        embedding_strategy=retrieval_strategy,
        candidates=DOCUMENT_TYPE_CANDIDATES,
    )

    return DocumentIntelligencePipeline(
        document_repository=document_repository,
        preparation_service=preparation_service,
        upload_dir=upload_dir,
        indexing_service=indexing_service,
        document_search_service=document_search_service,
        retrieval_service=retrieval_service,
        analysis_service=analysis_service,
        ocr_service=ocr_service,
        score_calibrator=visual_score_calibrator,
        text_indexing_service=text_indexing_service,
        classification_service=classification_service,
    )