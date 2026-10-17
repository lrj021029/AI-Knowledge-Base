from app.shared.model.lm_utils import get_llm_client
from app.shared.model.reranker_utils import get_reranker_model
from app.infra.config.providers import infra_config
from app.shared.model.embedding_utils import generate_embeddings

class LLMProvider():
    def chat(self, model:str | None=None, json_model:bool=False):
        return get_llm_client(model=model, json_mode=json_model)

    def vision_chat(self, model:str | None=None):
        if not model:
            model = infra_config.lm_config.lv_model
        return get_llm_client(model=model, json_mode=False)

    def generate_infra_embeddings(self, texts:list[str]) ->dict[str, list]:
        return generate_embeddings(texts)

    def reranker_computer_scores(self,question_answer_pair_list:list[list[str,str]]) -> list[float]:
        reranker_model = get_reranker_model()
        score_list:list[float] = reranker_model.compute_score(question_answer_pair_list,normalized=True)
        return score_list


    def reranker_str_to_tokens_number(self,content:str) -> int:
        reranker_model = get_reranker_model()
        tokenizer = reranker_model.tokenizer
        token_ids_list:list = tokenizer.encode(content)
        print(token_ids_list)
        token_numbers = len(token_ids_list)
        return token_numbers


llm_provider = LLMProvider()