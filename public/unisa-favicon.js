(function () {
  const faviconHref = "/favicon?v=unisa";
  const rels = ["icon", "shortcut icon", "apple-touch-icon"];

  document
    .querySelectorAll('link[rel="icon"], link[rel="shortcut icon"], link[rel="apple-touch-icon"]')
    .forEach((link) => link.remove());

  for (const rel of rels) {
    const link = document.createElement("link");
    link.rel = rel;
    link.type = "image/png";
    link.href = faviconHref;
    document.head.appendChild(link);
  }
})();
