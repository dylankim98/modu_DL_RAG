import numpy as np
from scipy.optimize import differential_evolution
from retriever import retriever


class AutoRanker:
    def __init__(self):
        # [w_ing, w_level, w_pop, w_style, p_time]
        self.params = np.array([2.0, 1.0, 0.5, 1.0, 1.0])

    def _level_score(self, level):
        if level in ["초급", "아무나", "쉬움", "Easy"]:
            return 1
        if level == "중급":
            return 0.5
        return 0

    def _parse_time(self, t):
        if "30" in str(t):
            return 0
        if "60" in str(t):
            return 1
        return 2

    # ------------------ scoring ------------------

    def score(self, doc, user_ings, style_hint):
        md = doc.metadata or {}
        text = doc.page_content or ""

        ing_hit = sum(1 for ing in user_ings if ing.strip() in text)
        level_score = self._level_score(md.get("level", ""))
        pop_score = np.log1p(int(md.get("views", 0)))

        style_score = 1 if style_hint and (
            style_hint in text
            or style_hint in str(md.get("method", ""))
            or style_hint in str(md.get("situation", ""))
        ) else 0

        time_pen = self._parse_time(md.get("time", ""))

        w_ing, w_level, w_pop, w_style, p_time = self.params

        return (
            w_ing * ing_hit
            + w_level * level_score
            + w_pop * pop_score
            + w_style * style_score
            - p_time * time_pen
        )

    # ------------------ objective ------------------

    def _objective(self, params, docs, user_ings, style_hint):
        self.params = params

        ranked = sorted(
            docs, key=lambda d: self.score(d, user_ings, style_hint), reverse=True
        )
        top = ranked[:5]

        views = np.mean([int(d.metadata.get("views", 0)) for d in top])

        ing_match = np.mean(
            [sum(1 for ing in user_ings if ing in (d.page_content or "")) for d in top]
        )

        style_match = np.mean(
            [
                1
                if style_hint
                and (
                    style_hint in (d.page_content or "")
                    or style_hint in str(d.metadata.get("situation", ""))
                    or style_hint in str(d.metadata.get("method", ""))
                )
                else 0
                for d in top
            ]
        )

        level_match = np.mean(
            [
                1
                if d.metadata.get("level") in ["초급", "아무나", "쉬움", "Easy"]
                else 0
                for d in top
            ]
        )

        final_score = (
            views
            + 2000 * ing_match
            + 1000 * style_match
            + 500 * level_match
        )

        return -final_score

    # ------------------ training ------------------

    def fit(self, docs, user_ings, style_hint):
        bounds = [(0, 5), (0, 5), (0, 2), (0, 3), (0, 3)]

        result = differential_evolution(
            self._objective,
            bounds=bounds,
            args=(docs, user_ings, style_hint),
            maxiter=40,
        )

        self.params = result.x
        return self.params


# ==================== RUN ====================

if __name__ == "__main__":

    user_story = "I want easy Korean snack"
    ingredients = "tofu, onion"
    style_hint = "간식"

    query = f"""
    User mood: {user_story}
    Ingredients: {ingredients}
    Style: {style_hint}
    Find suitable Korean recipes.
    Beginner friendly.
    """

    docs = retriever.invoke(query)
    user_ings = [i.strip() for i in ingredients.split(",")]

    ranker = AutoRanker()
    best_params = ranker.fit(docs, user_ings, style_hint)
    np.save("ranker_weights.npy", best_params)

    ranker.params = np.load("ranker_weights.npy")

    docs.sort(key=lambda d: ranker.score(d, user_ings, style_hint), reverse=True)

    print("\n🏆 TOP 5 MENUS\n")
    for d in docs[:5]:
        print(d.metadata["menu"], d.metadata["views"])
