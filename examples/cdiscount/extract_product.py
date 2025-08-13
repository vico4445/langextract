import os
import sys
import textwrap
import requests
import langextract as lx


def fetch_url_text(url: str, timeout: int = 30) -> str:
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/123.0.0.0 Safari/537.36"
      ),
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
      "Referer": "https://www.google.com/",
      "Connection": "keep-alive",
  }
  resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
  resp.raise_for_status()
  return resp.text


def load_input(arg: str) -> str:
  if arg == "-":
    return sys.stdin.read()
  if arg.startswith("http://") or arg.startswith("https://"):
    return fetch_url_text(arg)
  if os.path.exists(arg) and os.path.isfile(arg):
    with open(arg, "r", encoding="utf-8", errors="ignore") as f:
      return f.read()
  # Traiter comme texte brut passé en argument
  return arg


def main():
  if len(sys.argv) < 2:
    print("Usage: python extract_product.py <url|chemin_fichier|->")
    print("  -: lit le HTML depuis stdin")
    sys.exit(1)

  input_arg = sys.argv[1]

  # Prompt: extractions basiques pour une fiche e-commerce
  prompt = textwrap.dedent(
      """
      Extrait du texte fourni (qui peut contenir du HTML) les entités suivantes en respectant strictement le texte source (pas de paraphrase) et sans chevauchement d'indices.
      Ignore les balises HTML et le bruit de navigation.
      - product: nom complet du produit tel qu'affiché
      - brand: marque si identifiable
      - price: prix affiché (avec devise si présente)
      - rating: note moyenne (ex: 4,5/5 ou 4.5/5)
      - num_reviews: nombre d'avis si disponible
      - key_features: puces/caractéristiques principales (liste)
      - availability: disponibilité (ex: En stock, Rupture, Précommande)
      Ajoute des attributs pertinents pour chaque entité, par exemple devise pour price.
      """
  )

  examples = [
      lx.data.ExampleData(
          text="Apple iPhone 15 128 Go - Noir - 4,6/5 (123 avis) - 999,00 €",
          extractions=[
              lx.data.Extraction(
                  extraction_class="product",
                  extraction_text="Apple iPhone 15 128 Go - Noir",
                  attributes={"category": "smartphone"},
              ),
              lx.data.Extraction(
                  extraction_class="brand",
                  extraction_text="Apple",
              ),
              lx.data.Extraction(
                  extraction_class="price",
                  extraction_text="999,00 €",
                  attributes={"currency": "EUR"},
              ),
              lx.data.Extraction(
                  extraction_class="rating",
                  extraction_text="4,6/5",
              ),
              lx.data.Extraction(
                  extraction_class="num_reviews",
                  extraction_text="123 avis",
                  attributes={"count": 123},
              ),
          ],
      )
  ]

  try:
    source_text = load_input(input_arg)
  except Exception as e:
    print(f"Erreur lors du chargement de l'entrée: {e}")
    sys.exit(2)

  model_id = os.environ.get("LANGEXTRACT_MODEL", "gemini-2.5-flash")

  # Paramètres spécifiques OpenAI si utilisés
  fence_output = False
  use_schema_constraints = True
  if model_id.lower().startswith("gpt-"):
    fence_output = True
    use_schema_constraints = False

  try:
    result = lx.extract(
        text_or_documents=source_text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        max_char_buffer=1500,
        max_workers=10,
        extraction_passes=int(os.environ.get("LANGEXTRACT_PASSES", "1")),
        fence_output=fence_output,
        use_schema_constraints=use_schema_constraints,
    )
  except Exception as e:
    print("Erreur pendant l'extraction:")
    print(" ", e)
    print("Conseils: ")
    print(" - Vérifiez que la variable d'environnement LANGEXTRACT_API_KEY est définie pour Gemini/OpenAI")
    print(" - Essayez avec un autre modèle via LANGEXTRACT_MODEL (ex: gpt-4o si OPENAI_API_KEY est configurée)")
    print(" - Pour Ollama local: export LANGEXTRACT_MODEL=gemma2:2b et lancez ollama serve")
    sys.exit(3)

  out_jsonl = "cdiscount_product_extractions.jsonl"
  lx.io.save_annotated_documents([result], output_name=out_jsonl, output_dir=".")
  html = lx.visualize(out_jsonl)
  with open("cdiscount_product_visualization.html", "w", encoding="utf-8") as f:
    f.write(html)

  print("OK ->", out_jsonl, "cdiscount_product_visualization.html")


if __name__ == "__main__":
  main()