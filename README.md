# TriLens Document Intelligence

TriLens is een lokale multimodale document-intelligenceapplicatie voor visuele documentindexering, semantische retrieval, automatische captioning en experimentele documentanalyse.

De applicatie combineert:

- **CLIP** voor image-text retrieval;
- **BLIP** voor automatische documentcaptions;
- **OpenFlamingo** voor optionele vraaggestuurde documentanalyse;
- **FastAPI** als backend-API;
- **Next.js** als primaire gebruikersinterface;
- **Streamlit** als oorspronkelijke prototype-interface;
- **SQLite en NumPy** voor lokale metadata- en embeddingopslag.

> TriLens is een portfolio- en leerproject. Modeloutput kan onnauwkeurig zijn en mag niet worden gebruikt als juridisch, financieel of identiteitsadvies.

---

## Demo

De primaire interface bestaat uit één dashboard:

1. upload een documentafbeelding;
2. laat CLIP en BLIP het document verwerken;
3. zoek documenten met natuurlijke taal;
4. vergelijk CLIP- en hybride rankingscores;
5. voer rechtstreeks op een zoekresultaat een experimentele analyse uit.

### Unified dashboard

The primary Next.js interface combines document upload and semantic search in a single dashboard.

![TriLens dashboard with upload and search forms](docs/screenshots/01-dashboard.png)

### Semantic document retrieval

Search results include the document image, BLIP caption and individual ranking signals for CLIP, caption similarity and metadata similarity.

![TriLens hybrid document search results](docs/screenshots/02-search-results.png)

### Inline document analysis

A selected result can be analysed directly without navigating to a separate page. OpenFlamingo is optional, and TriLens can expose a BLIP caption fallback when analysis is unavailable.

![TriLens inline document analysis](docs/screenshots/03-inline-analysis.png)

### Document indexing

Uploaded images are validated, preprocessed, embedded with CLIP and captioned with BLIP.

![TriLens document indexing result](docs/screenshots/04-upload-result.png)

---

## Probleemstelling

Documentzoekmachines vertrouwen vaak op bestandsnamen, handmatig ingevoerde metadata of OCR-tekst.

TriLens onderzoekt een andere benadering: documenten zoeken op basis van hun visuele en semantische kenmerken.

Voorbeeldqueries:

```text
invoice with several product rows
store receipt
form with multiple input fields
document containing a signature
identity document
```

Hierdoor kunnen documenten ook worden teruggevonden wanneer de exacte woorden uit de query niet letterlijk in het document voorkomen.

---

## Architectuur

```text
Next.js dashboard
        │
        │ HTTP
        ▼
FastAPI API
        │
        ▼
DocumentIntelligencePipeline
        │
        ├── preprocessing
        │     ├── image validation
        │     ├── EXIF correction
        │     ├── RGB conversion
        │     └── resizing
        │
        ├── CLIP retrieval
        │     ├── image embeddings
        │     ├── text embeddings
        │     └── cosine similarity
        │
        ├── BLIP captioning
        │
        ├── hybrid reranking
        │
        └── optional OpenFlamingo analysis
              └── BLIP caption fallback
        │
        ▼
SQLite metadata + NumPy embeddings
```

De applicatie gebruikt een gedeelde application pipeline. FastAPI en Streamlit zijn dunne adapters rond dezelfde services en domeinlogica.

---

## CLIP, BLIP en OpenFlamingo

### CLIP

CLIP zet afbeeldingen en tekst om naar vectoren in een gedeelde embeddingruimte.

TriLens gebruikt CLIP voor:

- documentafbeeldingen indexeren;
- zoekqueries encoderen;
- cosine similarity berekenen;
- top-k-documenten rangschikken.

CLIP is geen OCR-systeem. Het is vooral geschikt voor visuele en semantische overeenkomsten.

### BLIP

BLIP genereert een korte beschrijving van een documentafbeelding.

De caption wordt:

- opgeslagen als modelartifact;
- weergegeven in zoekresultaten;
- gebruikt als extra signaal bij hybride ranking;
- gebruikt als fallback wanneer OpenFlamingo geen analyse kan leveren.

### OpenFlamingo

OpenFlamingo wordt experimenteel gebruikt om een vraag over één geselecteerd document te beantwoorden.

OpenFlamingo:

- staat standaard uit;
- wordt lazy geladen;
- kan zeer traag zijn op CPU;
- vereist veel geheugen;
- kan visuele details verkeerd interpreteren of hallucineren.

Op systemen met ongeveer 8 GB GPU-geheugen past de huidige configuratie mogelijk niet volledig in GPU-geheugen.

---

## Functionaliteiten

### Documentindexering

- PNG-, JPG- en JPEG-bestanden;
- bestand- en afbeeldingsvalidatie;
- SHA-256-checksum;
- detectie van eerder geïndexeerde documenten;
- preprocessing;
- CLIP-image-embedding;
- BLIP-caption;
- gedeeltelijk herstel wanneer één modelstap faalt;
- lokale opslag van metadata en artifacts.

### Zoeken

- natuurlijke-taalqueries;
- top-k-ranking;
- filteren op documenttype;
- CLIP-baseline;
- optionele hybride ranking;
- individuele scores voor:
  - CLIP;
  - captionovereenkomst;
  - metadataovereenkomst;
  - uiteindelijke ranking.

### Analyse

- vraaggestuurde analyse van één document;
- optionele OpenFlamingo-uitvoering;
- BLIP-captionfallback;
- modelnaam, bron en runtime in de response;
- waarschuwing voor onbetrouwbare modeloutput.

---

## Projectstructuur

```text
trilens-document-intelligence/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   ├── dependencies.py
│   │   └── main.py
│   ├── domain/
│   ├── preprocessing/
│   ├── repositories/
│   ├── services/
│   ├── strategies/
│   ├── ui/
│   └── bootstrap.py
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   └── lib/
│   ├── package.json
│   └── package-lock.json
├── tests/
├── data/
├── docs/
├── streamlit_app.py
├── pyproject.toml
└── README.md
```

Runtimebestanden, uploads, databases, embeddings en modelcaches horen niet in Git.

---

## Vereisten

Aanbevolen lokale omgeving:

- Python 3.12;
- Node.js 20 of nieuwer;
- npm;
- voldoende vrije schijfruimte voor modelbestanden;
- optioneel een CUDA-compatibele GPU.

OpenFlamingo vereist meerdere gigabytes aan modelbestanden en is niet nodig voor upload, captioning of retrieval.

---

## Installatie

Clone de repository:

```bash
git clone https://github.com/Johan-torfs/trilens-document-intelligence.git
cd trilens-document-intelligence
```

Maak een Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Installeer de Python-dependencies met de installatiemethode van het project:

```bash
pip install -e .
```

Wanneer het project nog geen installeerbare dependencyconfiguratie in `pyproject.toml` bevat, gebruik dan het bijgeleverde requirementsbestand:

```bash
pip install -r requirements.txt
```

Installeer de frontend:

```bash
cd frontend
npm ci
cd ..
```

---

## Configuratie

Kopieer de backendconfiguratie:

```bash
cp .env.example .env
```

Kopieer de frontendconfiguratie:

```bash
cp frontend/.env.example frontend/.env.local
```

### Backendvariabelen

```env
TRILENS_OPEN_FLAMINGO_ENABLED=false
TRILENS_OPEN_FLAMINGO_DEVICE=cpu
TRILENS_CORS_ORIGINS=http://localhost:3000
```

### Frontendvariabelen

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

OpenFlamingo blijft voor de standaard-MVP uitgeschakeld:

```env
TRILENS_OPEN_FLAMINGO_ENABLED=false
```

Experimentele CPU-analyse inschakelen:

```env
TRILENS_OPEN_FLAMINGO_ENABLED=true
TRILENS_OPEN_FLAMINGO_DEVICE=cpu
```

---

## Applicatie starten

### FastAPI

Start vanuit de projectroot:

```bash
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

De API is beschikbaar op:

```text
http://127.0.0.1:8000
```

Interactieve API-documentatie:

```text
http://127.0.0.1:8000/docs
```

Healthcheck:

```text
GET http://127.0.0.1:8000/api/health
```

### Next.js

Start in een tweede terminal:

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

### Streamlit-prototype

De oorspronkelijke prototype-interface blijft beschikbaar:

```bash
streamlit run streamlit_app.py
```

De Next.js-interface is de primaire portfoliofrontend.

---

## API-endpoints

| Methode | Endpoint                                | Beschrijving                         |
| ------- | --------------------------------------- | ------------------------------------ |
| `GET`   | `/api/health`                           | Controleert de API-status            |
| `POST`  | `/api/documents`                        | Uploadt en indexeert een document    |
| `POST`  | `/api/search`                           | Zoekt documenten                     |
| `GET`   | `/api/documents/{document_id}/image`    | Retourneert de documentafbeelding    |
| `POST`  | `/api/documents/{document_id}/analysis` | Analyseert een geselecteerd document |

---

## Voorbeeld: zoeken

Request:

```json
{
  "query": "invoice with several product rows",
  "top_k": 5,
  "document_type": "invoice",
  "use_hybrid_ranking": true
}
```

Vereenvoudigde response:

```json
{
  "ranking_mode": "hybrid",
  "results": [
    {
      "document_id": "example-document-id",
      "rank": 1,
      "final_score": 0.84,
      "clip_score": 0.79,
      "caption_score": 0.96,
      "metadata_score": 1.0,
      "caption": "an invoice containing multiple product rows",
      "image_url": "/api/documents/example-document-id/image",
      "document_type": "invoice"
    }
  ]
}
```

---

## Dataset

De huidige demonstratiedataset bevat synthetische, publieke en afgeleide documentafbeeldingen.

Documentcategorieën omvatten onder andere:

- facturen;
- purchase orders;
- kassabonnen;
- delivery notes;
- aanvraagformulieren;
- fictieve identiteitskaarten.

De dataset is klein en bedoeld voor architectuur- en functionaliteitsdemonstratie. Ze vormt geen representatieve productiebenchmark.

Gebruik geen echte identiteitsdocumenten, klantdocumenten of documenten met persoonsgegevens.

---

## Tests

Voer alle Python-tests uit:

```bash
python -m pytest
```

Voer de frontendcontroles uit:

```bash
cd frontend
npm run lint
npm run build
```

De testset bevat onder andere tests voor:

- afbeeldingsvalidatie;
- preprocessing;
- checksums;
- repositories;
- cosine similarity;
- ranking;
- CLIP-service-integratie;
- BLIP-captioning;
- OpenFlamingo-fallback;
- application pipeline;
- FastAPI-endpoints.

Modelafhankelijke tests gebruiken mocks waar mogelijk. CI hoort geen grote modelcheckpoints te downloaden.

---

## Privacy

TriLens is ontworpen als lokaal portfolio- en onderzoeksproject.

Belangrijke beperkingen:

- gebruik alleen synthetische, publieke of correct geanonimiseerde data;
- commit geen persoonsgegevens;
- commit geen echte identiteitsdocumenten;
- uploads worden lokaal verwerkt;
- de applicatie uploadt documenten niet automatisch naar een externe dienst;
- logging hoort geen afbeeldingsinhoud of gevoelige documenttekst te bevatten;
- modeloutput kan onjuist of verzonnen zijn;
- resultaten zijn geen juridisch, financieel of identiteitsadvies.

Controleer altijd de licentievoorwaarden van externe datasets voordat afbeeldingen worden gepubliceerd of herverdeeld.

---

## Bekende beperkingen

- De dataset is klein.
- Er is nog geen formele retrievalbenchmark.
- CLIP leest geen exacte documenttekst zoals een OCR-engine.
- Retrievalkwaliteit verschilt per documentcategorie en queryformulering.
- BLIP-captions zijn algemeen en missen soms kleine documentdetails.
- OpenFlamingo kan hallucineren of repetitieve output genereren.
- OpenFlamingo is langzaam op CPU.
- De huidige OpenFlamingo-configuratie kan een GPU met 8 GB geheugen overschrijden.
- Er is geen authenticatie of gebruikersbeheer.
- Er is geen rate limiting.
- Verwerking gebeurt synchroon.
- Alleen documentafbeeldingen worden ondersteund.
- Het systeem is niet bedoeld voor productiegebruik.

---

## Evaluatie

Een formele retrievalbenchmark maakt nog geen deel uit van de eerste MVP.

Een toekomstige evaluatie zal een vaste dataset en minimaal tien handmatig gelabelde queries gebruiken, met onder andere:

- Recall@1;
- Recall@3;
- gemiddelde querytijd;
- gemiddelde indexeringstijd;
- inhoudelijke foutanalyse.

De huidige dataset is primair bedoeld om de end-to-end-architectuur te demonstreren.

---

## Roadmap

### Eerstvolgende kwaliteitsfase

- aanvullende veilige documentdatasets onderzoeken;
- retrievalkwaliteit evalueren;
- modellen vergelijken en actualiseren;
- latency en geheugengebruik verbeteren;
- OpenFlamingo-prompts en checkpoints onderzoeken;
- CLIP- en captionreranking verfijnen.

### Mogelijke latere uitbreidingen

- automatische documentclassificatie;
- OCR en hybride text-image retrieval;
- ondersteuning voor PDF- en Office-documenten;
- documenten met meerdere pagina’s;
- asynchrone indexering;
- batchuploads;
- Docker en persistente modelcachevolumes;
- uitgebreidere observability;
- productie-authenticatie en rate limiting.

---

## Technische keuzes

### Waarom lokale opslag?

SQLite en NumPy houden de MVP:

- eenvoudig;
- inspecteerbaar;
- lokaal;
- reproduceerbaar;
- vrij van externe infrastructuur.

### Waarom een application pipeline?

`DocumentIntelligencePipeline` orkestreert de gespecialiseerde services zonder model-, opslag- of UI-logica te dupliceren.

Hierdoor gebruiken FastAPI en Streamlit dezelfde kernfunctionaliteit.

### Waarom twee frontends?

Streamlit werd gebruikt om de ML-flow snel te valideren.

Daarna werd een gescheiden FastAPI- en Next.js-architectuur toegevoegd om een realistischer applicatieontwerp te demonstreren.

---

## Status

**MVP 1**

Werkend:

- documentupload;
- preprocessing;
- CLIP-indexering;
- BLIP-captioning;
- semantische retrieval;
- hybride ranking;
- optionele OpenFlamingo-analyse;
- captionfallback;
- lokale opslag;
- FastAPI;
- Next.js-dashboard;
- Streamlit-prototype;
- geautomatiseerde tests.

Gepland na MVP 1:

- formele evaluatie;
- aanvullende datasets;
- modelkwaliteitsverbetering;
- performanceoptimalisatie;
- automatische classificatie;
- ondersteuning voor andere documentformaten;
- aanvullen met ocr-enigne om documenten inhoudeljk te analyzeren.

---

## Licentie

De eigen broncode en projectdocumentatie van TriLens Document Intelligence worden beschikbaar gesteld onder de MIT License.

Zie [LICENSE](LICENSE) voor de volledige licentietekst.

Deze licentie geldt niet automatisch voor:

- externe modelweights;
- externe datasets;
- afbeeldingen uit externe datasets;
- softwaredependencies;
- code of assets van derden.

CLIP-, BLIP- en OpenFlamingo-modellen en hun checkpoints behouden hun eigen licentievoorwaarden. Hetzelfde geldt voor Hugging Face-datasets en andere publieke databronnen.

Gebruikers en bijdragers zijn zelf verantwoordelijk voor het controleren van de toepasselijke model-, dataset- en dependencylicenties voordat zij bestanden herverdelen, publiceren of commercieel gebruiken.
