from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from html_utils import resolve_href  # noqa: E402


class HtmlUtilsTests(unittest.TestCase):
    def test_rescue_relative_course_menu_link_resolves_to_course_root(self) -> None:
        base_url = (
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/"
            "url/L2luZ2VnbmVyaWEtZGVsbC1pbmZvcm1hemlvbmUtcGVyLWxhLW1lZGljaW5hLWRpZ2l0YWxl/"
            "id/1540/module/501/row/29862"
        )

        self.assertEqual(
            resolve_href(base_url, "contatti"),
            "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale/contatti",
        )

    def test_rescue_relative_numeric_course_link_resolves_from_site_root(self) -> None:
        base_url = (
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/"
            "url/LzA2NTAxMDYyMDA4MDAwMDM%3D/id/1540/module/501/row/29862"
        )

        self.assertEqual(
            resolve_href(base_url, "0650106200800003/contatti"),
            "https://corsi.unisa.it/0650106200800003/contatti",
        )

    def test_numeric_course_alias_link_resolves_from_site_root(self) -> None:
        self.assertEqual(
            resolve_href(
                "https://corsi.unisa.it/0650107303300001/attivita-e-servizi",
                "0650107303300001/contatti",
            ),
            "https://corsi.unisa.it/0650107303300001/contatti",
        )

    def test_rescue_relative_plain_link_with_encoded_padding_uses_decoded_course(self) -> None:
        base_url = (
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/"
            "url/LzA2NTAxMDYyMDA4MDAwMDM%3D/id/1540/module/501/row/29862"
        )

        self.assertEqual(
            resolve_href(base_url, "contatti"),
            "https://corsi.unisa.it/0650106200800003/contatti",
        )

    def test_rescue_search_relative_course_menu_link_resolves_to_course_root(self) -> None:
        base_url = (
            "https://corsi.unisa.it/unisa-rescue-page/search/id/2364/"
            "url/LzA2MjI3L2VuL3RlYWNoaW5nLWZhY2lsaXRpZXM=/"
            "calendario-occupazione-spazi/calendario-occupazione-spazi"
        )

        self.assertEqual(
            resolve_href(base_url, "0650107303300001/contatti"),
            "https://corsi.unisa.it/0650107303300001/contatti",
        )


if __name__ == "__main__":
    unittest.main()
