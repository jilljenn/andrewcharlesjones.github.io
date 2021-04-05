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

    <!-- <li itemscope>
      <a href="{{ site.github.url }}{{ post.url }}" style="text-decoration:none">{{ post.title }}</a>
      <p class="post-blurb">{{ post.blurb }}</p>
      <p class="post-date"><span> {{ post.date | date: "%B %-d" }}</span></p>
    </li> -->
    <li itemscope>
      <div>
        <a href="{{ site.github.url }}{{ post.url }}" style="text-decoration:none">{{ post.title }}</a>
        <span class="post-date"> {{ post.date | date: "%B %-d" }}</span>
        <img src="{{ post.img }}" align="right" width="150">
        <p class="post-blurb">{{ post.blurb }}</p>
      </div>
    </li>

  {% endfor %}
</ul>
