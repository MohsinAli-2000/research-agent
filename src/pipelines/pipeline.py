from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
    extract_text_content,
)

def run_research_pipeline(topic : str) -> dict:

    state = {}

    # Step 1: Search Agent
    print("\n" + " ="*50)
    print("Step 1 - Search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = extract_text_content(search_result['messages'][-1].content)

    print("\nSearch results:\n", state['search_results'])


    # Step 2: Reader Agent
    print("\n" + " ="*50)
    print("Step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:1000]}"
        )]
    })

    state['scraped_content'] = extract_text_content(reader_result['messages'][-1].content)

    print("\nScraped content:\n", state['scraped_content'])


    # Step 3: Writer Chain
    print("\n" + " ="*50)
    print("Step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = extract_text_content(writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    }))

    print("\nFinal Report:\n", state['report'])


    # Step 4: Critic Chain
    print("\n" + " ="*50)
    print("Step 4 - Critic is reviewing the report ...")
    print("="*50)

    state["feedback"] = extract_text_content(critic_chain.invoke({
        "report": state['report']
    }))

    print("\nCritic Report:\n", state['feedback'])

    return state
