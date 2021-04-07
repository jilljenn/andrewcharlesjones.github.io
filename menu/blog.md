---
layout: page
title: 
---
<ul class="posts">
  {% for post in site.posts %}

    {% unless post.next %}
      <h3>{{ post.date | date: '%Y' }}</h3>
    {% else %}
      {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
      {% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}
      {% if year != nyear %}
        <h3>{{ post.date | date: '%Y' }}</h3>
      {% endif %}
    {% endunless %}
    <li itemscope>
      <div style="margin-top:10%;">
        <a href="{{ site.github.url }}{{ post.url }}" style="text-decoration:none;">{{ post.title }}</a>
        <span class="post-date"> {{ post.date | date: "%B %-d" }}</span>
        {% if post.img != "" %}
          <img src="{{ post.img }}" align="right" width="150">
        {% else %}
          <img src="/assets/dice-six-solid.svg" align="right" width="40" style="opacity: 0.5;">
        {% endif %}
        <p class="post-date">{{ post.blurb }}</p>
      </div>
    </li>
  {% endfor %}
</ul>
