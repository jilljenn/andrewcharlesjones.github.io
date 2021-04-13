---
layout: page
title: 
---
These posts are informal notes and may contain errors --- please let me know if you spot any.

<ul class="posts">
  {% for post in site.posts %}
    <div style="margin-top:10%;">
    {% unless post.next %}
      <h2>{{ post.date | date: '%Y' }}</h2>
    {% else %}
      {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
      {% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}
      {% if year != nyear %}
        <h2>{{ post.date | date: '%Y' }}</h2>
      {% endif %}
    {% endunless %}
  </div>
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
