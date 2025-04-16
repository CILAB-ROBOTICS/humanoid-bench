

def is_renderable(env):
    try:
        env.render()
        return True
    except Exception:
        return False